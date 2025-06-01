import json
import multiprocessing

from ralm.retrievers.base_retrieval import BaseRetriever
from pyserini.search.faiss import FaissSearcher
from eval_qa import TELEGRAM_HANDLER
from transformers import AutoTokenizer, AutoModel


class DenseRetriever(BaseRetriever):
    def __init__(self, tokenizer, index_name, args=None):
        self.tokenizer = AutoTokenizer.from_pretrained(args.embedder_name)
        super(DenseRetriever, self).__init__(tokenizer=self.tokenizer)
        self.searcher = self._get_searcher(index_name, args.embedder_name)


    def _get_searcher(self, index_name, embedder_name):
        try:
            print("Attempting to treat the index as a directory (not prebuilt by pyserini)")
            searcher = FaissSearcher(
                index_name,
                embedder_name
            )

        except Exception as e:
            print(e)
            print("The index doesn't exist in the local OS")
            print(f"Attempting to download the index as if prebuilt by pyserini")
            searcher = FaissSearcher.from_prebuilt_index(
                index_name,
                embedder_name
            )
            TELEGRAM_HANDLER.emit("loaded prebuilt index from pyserini")
            print("loaded prebuilt index from pyserini")
        assert searcher is not None, "searcher is None - Critical error"

        return searcher


    def retrieve(self, dataset, k=1):
        # Step 1: Extract queries
        queries = dataset["question"]

        # Step 2: Perform batch search
        all_res = self.searcher.batch_search(
            queries,
            q_ids=[str(i) for i in range(len(queries))],
            k=k,
            threads=multiprocessing.cpu_count()
        )

        # Step 3: Create list of retrieved documents per query
        retrieved_docs_list = []
        for i in range(len(dataset)):
            res = all_res[str(i)]
            allowed_docs = []
            for hit in res[:k]:
                res_dict = json.loads(self.searcher.doc(hit.docid).raw())
                content_str = res_dict["contents"]
                allowed_docs.append({
                    "text": content_str,
                    "score": str(hit.score),
                    "doc_id": str(hit.docid)
                })
            retrieved_docs_list.append(allowed_docs)

        # Step 4: Add retrieved docs and queries as new columns
        dataset = dataset.add_column("query", queries)
        dataset = dataset.add_column("retrieved_docs", retrieved_docs_list)

        return dataset