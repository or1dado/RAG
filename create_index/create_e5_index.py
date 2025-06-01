import os
import faiss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import Tensor
import torch.nn.functional as F
from eval_qa import TELEGRAM_HANDLER
from transformers import AutoModel, AutoTokenizer

# Use DataParallel across GPU 0 and 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def encode_batch(passages, model, tokenizer, batch_size=32):
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

        # Move to device
        inputs = {k: v.to(DEVICE)
                  for k, v in batch_tokens.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            # Average pooling
            def average_pool(hidden_states: Tensor, mask: Tensor) -> Tensor:
                hidden_states = hidden_states.masked_fill(~mask[..., None].bool(), 0.0)
                summed = hidden_states.sum(dim=1)
                counts = mask.sum(dim=1)[..., None]
                out = summed / counts
                del summed, counts
                return out

            embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Move to CPU immediately and convert to numpy
            embeddings_cpu = embeddings.cpu().numpy().astype('float32')
            all_embeddings.append(embeddings_cpu)

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
        # model-specific prefix
        if ((isinstance(model,torch.nn.DataParallel) and model.module.config.name_or_path == "intfloat/e5-large-v2") or
                (model.config.name_or_path == "intfloat/e5-large-v2")):
            texts = ["passage: " + t for t in texts if t != "text"]

        embeddings = encode_batch(texts, model, tokenizer)
        index.add(embeddings)

        df.to_json(metadata_path, orient='records', lines=True, force_ascii=False, mode='a')

        if chunk_id % 100 == 0:
            TELEGRAM_HANDLER.emit(f"<b>Finish Chunk:</b> {chunk_id}")

    # Save index files
    TELEGRAM_HANDLER.emit("<b>Save Index...</b>")
    faiss.write_index(index, index_paths['Flat'])

    TELEGRAM_HANDLER.emit("<b>Finish Create Index</b>")


if __name__ == '__main__':
    model_name = 'intfloat/e5-large-v2'
    model, tokenizer = initialize_embedding_model(model_name, parallel=False)

    data_dir = '/lv_local/home/or.dado/PycharmProjects/RAG/create_index/downloads/data/wikipedia_split'
    tsv_path = os.path.join(data_dir, 'psgs_w100.tsv')
    meta_path = os.path.join(data_dir, 'psgs_w100_metadata_E5.jsonl')
    idx_paths = {'Flat': os.path.join(data_dir, 'psgs_w100_E5_FLAT.index')}

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
        chunk_size=1024
    )