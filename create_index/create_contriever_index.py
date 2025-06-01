import os
import faiss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from eval_qa import TELEGRAM_HANDLER
from transformers import AutoModel, AutoTokenizer

# Use DataParallel across GPU 0 and 1
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def initialize_embedding_model(model_name, parallel=False):
    """
    Initialize the Contriever model and tokenizer, wrap model in DataParallel for multi-GPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    # Wrap model for multi-GPU and move to primary device
    if parallel:
        model = torch.nn.DataParallel(base_model, device_ids=[0, 1])
    else:
        model = base_model
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def encode_batch(passages, model, tokenizer, batch_size=64):
    """
    Process passages in smaller sub-batches to avoid memory issues.
    """
    all_embeddings = []

    # Process in smaller batches
    for i in range(0, len(passages), batch_size):
        batch_passages = passages[i:i + batch_size]

        # Tokenize batch
        batch_tokens = tokenizer(
            batch_passages,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Move input tensors to the same device as the model
        inputs = {key: val.to(DEVICE) for key, val in batch_tokens.items()}

        # Compute token embeddings
        with torch.no_grad():
            outputs = model(**inputs)

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


def create_faiss_index(
    csv_path,
    metadata_path,
    index_paths: dict,
    model,
    tokenizer,
    chunk_size=2000
):
    # Dimension and FAISS index init
    dim = model.module.config.hidden_size if isinstance(model, torch.nn.DataParallel) else model.config.hidden_size
    index = faiss.IndexFlatIP(dim)

    dtype_spec = {
        'id': str,
        'text': str,
        'title': str
    }

    total_lines = 21015325
    total_chunks = total_lines // chunk_size + (1 if total_lines % chunk_size != 0 else 0)

    # Iterate through chunks
    for chunk_id, df in enumerate(tqdm(
            pd.read_csv(csv_path, sep='\t', names=['id', 'text', 'title'], dtype=dtype_spec, chunksize=chunk_size, header=0),
            total=total_chunks,
            desc="Embedding & Indexing"), start=1):

        texts = df['text'].astype(str).tolist()
        texts = [t for t in texts if t not in ["id", "text", "title"]]

        embeddings = encode_batch(texts, model, tokenizer)
        index.add(embeddings)

        df.to_json(metadata_path, orient='records', lines=True, force_ascii=False, mode='a')

        if chunk_id % 100 == 0:
            TELEGRAM_HANDLER.emit(f"<b>Finish Chunk:</b> {chunk_id}")

    Save index files
    TELEGRAM_HANDLER.emit("<b>Save Index...</b>")
    faiss.write_index(index, index_paths['Flat'])

    TELEGRAM_HANDLER.emit("<b>Finish Create Index</b>")


if __name__ == '__main__':
    model_name = 'facebook/contriever'
    model, tokenizer = initialize_embedding_model(model_name, parallel=False)

    data_dir = '/lv_local/home/or.dado/PycharmProjects/RAG/create_index/downloads/data/wikipedia_split'
    tsv_path = os.path.join(data_dir, 'psgs_w100.tsv')
    meta_path = os.path.join(data_dir, 'psgs_w100_metadata_contriever.jsonl')
    idx_paths = {'Flat': os.path.join(data_dir, 'psgs_w100_contriever_FLAT.index')}

    TELEGRAM_HANDLER.emit(
        f"###########################\n"
        f"<b>Start Create Faiss Index</b>\n"
        f"<b>Dataset:</b> {os.path.basename(data_dir)}\n"
        f"<b>Embedder Name:</b> {model_name}\n"
    )

    create_faiss_index(
        tsv_path,
        meta_path,
        idx_paths,
        model,
        tokenizer,
        chunk_size=4096
    )