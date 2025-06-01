import gc
import time
import json
import faiss
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing
import torch.nn.functional as F
from eval_qa import TELEGRAM_HANDLER
from ralm.retrievers.base_retrieval import BaseRetriever
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


class DenseRetriever(BaseRetriever):
    def __init__(self, tokenizer, index_name, args=None):
        self.tokenizer = AutoTokenizer.from_pretrained(args.embedder_name)
        super(DenseRetriever, self).__init__(tokenizer=self.tokenizer)
        self.index_name = index_name
        self.searcher = self._get_searcher(index_name)
        self.encoder = SentenceTransformer(args.embedder_name, device = 'cuda') \
            if args.embedder_name == "intfloat/e5-large-v2" else (
            AutoModel.from_pretrained(args.embedder_name))
        self.document_lookup = self._load_document_lookup(args.docid_mapping)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rerank_docs = args.ranking_strategy == "rerank"
        self.args = args


    def _load_document_lookup(self, docid_mapping_path):
        # Load document mapping from docid to actual document content
        raw_passages = []
        progress_bar = tqdm(total=21015324, ncols=100, desc='loading docid mapping...')
        with open(docid_mapping_path, 'r') as file:
            for i, line in enumerate(file):
                raw_passages.append(line)
                progress_bar.update(1)

        self.lookup_len = len(raw_passages)
        TELEGRAM_HANDLER.emit("Finished loading DocID mapping.")
        return raw_passages


    def _get_searcher(self, index_name):
        try:
            print(f"Loading FAISS index from {index_name}")
            start_time = time.time()
            index = faiss.read_index(index_name)
            TELEGRAM_HANDLER.emit("Successfully Loaded FAISS Index")
            print(f"Successfully loaded FAISS index in {time.time() - start_time:.2f}s")
            return index

        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            raise ValueError(f"Failed to load FAISS index from '{index_name}'")


    def _contriever_encode(self, queries, batch_size=32):
        self.encoder.to(self.device)
        self.encoder.eval()
        all_embeddings = []

        # Process in smaller batches
        for i in range(0, len(queries), batch_size):
            batch_passages = queries[i:i + batch_size]

            # Tokenize batch
            batch_tokens = self.tokenizer(
                batch_passages,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            # Move input tensors to the same device as the model
            inputs = {key: val.to(self.device) for key, val in batch_tokens.items()}

            # Compute token embeddings
            with torch.no_grad():
                outputs = self.encoder(**inputs)

            # Mean pooling
            def mean_pooling(token_embeddings, attention_mask):
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
                sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask

            # Apply mean pooling
            embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

            # Cleanup GPU memory after each batch
            del batch_tokens, inputs, outputs, embeddings

            # cleanup
            torch.cuda.empty_cache()

        # Concatenate all embeddings
        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])


    def _E5_encode(self, queries, batch_size=32):
        if not queries:
            return np.array([])

        # Prefix queries as required by E5 models
        prefixed_queries = ["query: " + str(q) for q in queries]

        # Encode queries using the model's encode method
        embeddings = self.encoder.encode(
            prefixed_queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device='cuda',
        )

        # Optional cleanup: free memory
        del prefixed_queries
        gc.collect()
        torch.cuda.empty_cache()

        return embeddings


    def _encode_queries(self, queries, batch_size=32):
        model_name = self.args.embedder_name

        if "contriever" in model_name:
            return self._contriever_encode(queries, batch_size=batch_size)
        elif "intfloat/e5-large-v2" in model_name:
            return self._E5_encode(queries, batch_size=batch_size)
        else:
            raise ValueError(f"Encoder '{model_name}' not supported")


    def rerank(self, queries, retrieved_docs_list):
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
        model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
        model.eval()
        model.to(self.device)

        rerank_list = []
        for query, retrieved_docs in zip(queries, retrieved_docs_list):
            pairs = [[query, retrieved_doc["text"]] for retrieved_doc in retrieved_docs]

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt').to(self.device)
                scores = model(**inputs, return_dict=True).logits.view(-1).float()

            sorted_indices = torch.argsort(scores, descending=True).tolist()
            sorted_docs = [retrieved_docs[i] for i in sorted_indices]
            rerank_list.append(sorted_docs)

        return rerank_list


    def retrieve(self, dataset, k=1):
        # Set parallelism for FAISS
        n_cpus = multiprocessing.cpu_count()
        faiss.omp_set_num_threads(n_cpus)

        # Step 1: Extract queries
        queries = dataset["question"]

        # Step 2: Encode queries
        start_time = time.time()
        TELEGRAM_HANDLER.emit("Getting Queries Vector...")
        query_embeddings = self._encode_queries(queries)
        print(f"Getting queries vector in {time.time() - start_time:.2f}s")

        # Step 3: FAISS search
        TELEGRAM_HANDLER.emit("Starting Retrieval...")
        distances, indices = self.searcher.search(query_embeddings, k)

        # Step 4: Prepare retrieved_docs list
        all_retrieved_docs = []
        for distances_i, indices_i in zip(distances, indices):
            allowed_docs = []
            for distance, doc_idx in zip(distances_i, indices_i):
                score = float(distance)
                doc_id = int(doc_idx)

                if doc_id < self.lookup_len:
                    meta_data = json.loads(self.document_lookup[doc_id])
                    content_str = f"Title: {meta_data['title']}\n{meta_data['text']}"
                    allowed_docs.append({
                        "text": content_str,
                        "title": meta_data["title"],
                        "score": str(score),
                        "doc_id": doc_id
                    })
                else:
                    print(f"Document {doc_idx} not found!")

                if len(allowed_docs) >= k:
                    break

            all_retrieved_docs.append(allowed_docs)

        # Step 6: rerank retrieved doc
        if self.rerank_docs:
            TELEGRAM_HANDLER.emit("Start reranking...")
            all_retrieved_docs = self.rerank(queries, all_retrieved_docs)

        # Step 7: Add columns to dataset
        dataset = dataset.add_column("query", queries)
        dataset = dataset.add_column("retrieved_docs", all_retrieved_docs)

        del self.encoder
        torch.cuda.empty_cache()

        return dataset
