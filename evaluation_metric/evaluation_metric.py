import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder, SentenceTransformer

from eval_qa import normalize_answer


device = "cuda" if torch.cuda.is_available() else "cpu"


def init_semantic_ce_model(model_name):
    return CrossEncoder(model_name).to(device)


def _calculate_semantic_sim_ce(row, model):
    pred = normalize_answer(row["prediction"].split("\n\n")[-1])
    pairs = [(normalize_answer(gt_opt), pred) for gt_opt in row["answer"]]
    scores = model.predict(pairs)

    return scores

def get_semantic_similarity_ce(df, model):
    df["semantic_similarity_scores_ce"] = df.progress_apply(lambda row: _calculate_semantic_sim_ce(row, model), axis=1)
    df["max_semantic_similarity_ce"] = df["semantic_similarity_scores_ce"].apply(max)
    df["gt_targeted_ce"] = df.apply(
        lambda row: row["answer"][np.argmax(row["semantic_similarity_scores_ce"])],
        axis=1
    )

    return df


def init_e5_model(model_name):
    return SentenceTransformer(model_name)


def _calculate_e5_similarity_sbert(row, model, batch_size=32):
    # Normalize and prefix prediction and answers as required by E5
    pred = "query: " + normalize_answer(row["prediction"].split("\n\n")[-1])
    gt = [normalize_answer(ans) for ans in row["answer"]]

    # E5 expects everything to be prefixed with "query: "
    prefixed_passages = [pred] + ["query: " + ans for ans in gt]

    # Encode all inputs with normalization and GPU
    embeddings = model.encode(
        prefixed_passages,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
        show_progress_bar=False
    )

    # Compute cosine similarity between prediction (embeddings[0]) and each ground truth
    scores = embeddings[0] @ embeddings[1:].T

    return scores

def get_semantic_similarity_e5(df, model):
    df["semantic_similarity_scores_e5"] = df.progress_apply(lambda row: _calculate_e5_similarity_sbert(row, model), axis=1)
    df["max_semantic_similarity_e5"] = df["semantic_similarity_scores_e5"].apply(max)
    df["gt_targeted_e5"] = df.apply(
        lambda row: row["answer"][np.argmax(row["semantic_similarity_scores_e5"])],
        axis=1
    )

    return df

def init_dpr_model(model_name):
    from transformers import DPRContextEncoder

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DPRContextEncoder.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return model, tokenizer


def _calculate_dpr_similarity_bi(row, model, tokenizer):
    pred = normalize_answer(row["prediction"].split("\n\n")[-1])
    gt = [normalize_answer(ans) for ans in row["answer"]]
    prefixed_passages = [pred] + gt

    # Tokenize with padding and batching
    inputs = tokenizer(prefixed_passages, return_tensors="pt", padding=True)

    # Move input tensors to GPU
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Run model on GPU
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output  # shape: (batch_size, hidden_size)

    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.cpu().numpy()
    scores = embeddings[0] @ embeddings[1:].T

    return scores


def get_semantic_similarity_dpr(df, model, tokenizer):
    df["semantic_similarity_scores_dpr"] = df.progress_apply(lambda row: _calculate_dpr_similarity_bi(row, model, tokenizer), axis=1)
    df["max_semantic_similarity_dpr"] = df["semantic_similarity_scores_dpr"].apply(max)
    df["gt_targeted_dpr"] = df.apply(
        lambda row: row["answer"][np.argmax(row["semantic_similarity_scores_dpr"])],
        axis=1
    )

    return df


def init_contriever_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return model, tokenizer


def _calculate_contriever_similarity_bi(row, model, tokenizer):
    pred = normalize_answer(row["prediction"].split("\n\n")[-1])
    gt = [normalize_answer(ans) for ans in row["answer"]]
    prefixed_passages = [pred] + gt

    inputs = tokenizer(prefixed_passages, return_tensors="pt", padding=True)

    # Move input tensors to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

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
    scores = embeddings[0] @ embeddings[1:].T

    return scores

def init_minilm_model(model_name):
    return SentenceTransformer(model_name)


def _calculate_minilm_similarity_sbert(row, model, batch_size=32):
    # Normalize and prefix prediction and answers as required by E5
    pred = normalize_answer(row["prediction"].split("\n\n")[-1])
    gt = [normalize_answer(ans) for ans in row["answer"]]

    prefixed_passages = [pred] + [ans for ans in gt]

    # Encode all inputs with normalization and GPU
    embeddings = model.encode(
        prefixed_passages,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
        show_progress_bar=False
    )

    # Compute cosine similarity between prediction (embeddings[0]) and each ground truth
    scores = embeddings[0] @ embeddings[1:].T

    return scores

def get_semantic_similarity_minilm(df, model):
    df["semantic_similarity_scores_minilm"] = df.progress_apply(lambda row: _calculate_minilm_similarity_sbert(row, model), axis=1)
    df["max_semantic_similarity_minilm"] = df["semantic_similarity_scores_minilm"].apply(max)
    df["gt_targeted_minilm"] = df.apply(
        lambda row: row["answer"][np.argmax(row["semantic_similarity_scores_minilm"])],
        axis=1
    )

    return df

def get_semantic_similarity_contriever(df, model, tokenizer):
    df["semantic_similarity_scores_contriever"] = df.progress_apply(lambda row: _calculate_contriever_similarity_bi(row, model, tokenizer), axis=1)
    df["max_semantic_similarity_contriever"] = df["semantic_similarity_scores_contriever"].apply(max)
    df["gt_targeted_contriever"] = df.apply(
        lambda row: row["answer"][np.argmax(row["semantic_similarity_scores_contriever"])],
        axis=1
    )

    return df


def bert_score(row, bertscore):
    pred = normalize_answer(row["prediction"].split("\n\n")[-1])
    pairs = [(normalize_answer(gt_opt), pred) for gt_opt in row["answer"]]

    predictions = [p[1] for p in pairs]
    references = [p[0] for p in pairs]

    result = bertscore.compute(predictions=predictions, references=references, lang="en")

    return max(result["precision"]), max(result["recall"]), max(result["f1"])


def get_bert_score(df):
    from evaluate import load
    bertscore = load("bertscore")

    df[["max_semantic_similarity_precision_bert", "max_semantic_similarity_recall_bert",
        "max_semantic_similarity_f1_bert"]] = (
        df.progress_apply(bert_score, args=(bertscore,), axis=1, result_type="expand"))

    return df


def calculate_gain_contribution(rag_semantic_similarity, no_rag_semantic_similarity):
    eps = 10**-10
    return np.log2((rag_semantic_similarity+eps) / (no_rag_semantic_similarity+eps))
    # return (rag_semantic_similarity+eps) / (no_rag_semantic_similarity+eps)


def plot_corr_map(df, title, method):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Define column order and grouping
    ordered_columns = [
        "gain_bertscore_P", "gain_bertscore_R", "gain_bertscore_f1",  # BertScore
        "gain_contriever", "gain_dpr",                                # bi-encoder
        "gain_ce",                                                    # CE
        "gain_e5", "gain_minilm"                                      # Sbert
    ]

    high_level = [
        "BertScore", "BertScore", "BertScore",
        "bi-encoder", "bi-encoder",
        "CE",
        "Sbert", "Sbert"
    ]

    # Create MultiIndex for better x/y tick labeling
    multi_index = pd.MultiIndex.from_tuples(
        list(zip(high_level, ordered_columns)),
        names=["Model", "Metric"]
    )

    # Compute correlation
    corr_df = df[ordered_columns].corr(method=method)

    # Reindex rows and columns
    corr_df.columns = multi_index
    corr_df.index = multi_index

    # Plot
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        corr_df,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        linewidths=0.5,
        vmin=0,
        vmax=1
    )

    # Rotate tick labels for clarity
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Correlation Matrix of Gain Contributions - {title}\nMethod: {method}")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ################
    # Load Data    #
    ################
    no_rag_df = pd.read_pickle(r"/lv_local/home/or.dado/PycharmProjects/RAG/output/metric_results/gain/Llama_nqopen_500samples_without_rag.pkl")
    # rag_df = pd.read_pickle(r"/lv_local/home/or.dado/PycharmProjects/RAG/output/metric_results/gain/Llama_nqopen_500samples_dpr_20docs_rag.pkl")
    # rag_df = pd.read_pickle(r"/lv_local/home/or.dado/PycharmProjects/RAG/output/metric_results/gain/Llama_nqopen_500samples_contriever_20docs_rag.pkl")
    rag_df = pd.read_pickle(r"/lv_local/home/or.dado/PycharmProjects/RAG/output/metric_results/gain/Llama_nqopen_500samples_E5_20docs_rag.pkl")

    ##################################
    # semantic similarity measure    #
    ##################################

    # Initialize all models
    e5_model = init_e5_model(model_name="intfloat/e5-large-v2")
    ce_model = init_semantic_ce_model(model_name="cross-encoder/stsb-roberta-large")
    minilm_model = init_e5_model(model_name="sentence-transformers/all-MiniLM-L6-v2")
    dpr_model, dpr_tokenizer = init_dpr_model("facebook/dpr-ctx_encoder-multiset-base")
    contriever_model, contriever_tokenizer = init_contriever_model("facebook/contriever")

    # Process no_rag_df - build up columns progressively
    print("Processing no_rag_df...")
    no_rag_df = get_bert_score(no_rag_df)
    no_rag_df = get_semantic_similarity_ce(no_rag_df, ce_model)
    no_rag_df = get_semantic_similarity_e5(no_rag_df, e5_model)
    no_rag_df = get_semantic_similarity_minilm(no_rag_df, minilm_model)
    no_rag_df = get_semantic_similarity_dpr(no_rag_df, dpr_model, dpr_tokenizer)
    no_rag_df = get_semantic_similarity_contriever(no_rag_df, contriever_model, contriever_tokenizer)

    # Select final columns for no_rag_df
    no_rag_df = no_rag_df[[
        "question", "prediction", "max_semantic_similarity_precision_bert", "max_semantic_similarity_recall_bert",
        "max_semantic_similarity_f1_bert", "gt_targeted_ce", "gt_targeted_e5", "gt_targeted_dpr",
        "gt_targeted_contriever", "max_semantic_similarity_ce", "max_semantic_similarity_e5",
        "max_semantic_similarity_dpr", "max_semantic_similarity_contriever", "max_semantic_similarity_minilm", "EM"
    ]]

    # Process rag_df - build up columns progressively
    print("Processing rag_df...")
    rag_df = get_bert_score(rag_df)
    rag_df = get_semantic_similarity_ce(rag_df, ce_model)
    rag_df = get_semantic_similarity_e5(rag_df, e5_model)
    rag_df = get_semantic_similarity_minilm(rag_df, minilm_model)
    rag_df = get_semantic_similarity_dpr(rag_df, dpr_model, dpr_tokenizer)
    rag_df = get_semantic_similarity_contriever(rag_df, contriever_model, contriever_tokenizer)

    # Select final columns for rag_df
    rag_df = rag_df[[
        "question", "answer", "prediction", "max_semantic_similarity_precision_bert",
        "max_semantic_similarity_recall_bert", "max_semantic_similarity_f1_bert",
        "gt_targeted_ce", "gt_targeted_e5", "gt_targeted_dpr", "gt_targeted_contriever",
        "max_semantic_similarity_ce", "max_semantic_similarity_e5",
        "max_semantic_similarity_dpr", "max_semantic_similarity_contriever", "max_semantic_similarity_minilm", "EM"
    ]]

    # Merge dataframes
    print("Merging dataframes...")
    df = rag_df.merge(no_rag_df, on=["question"], how="inner", suffixes=('_rag', '_no_rag'))

    # Calculate gains
    print("Calculating gains...")
    df["gain_bertscore_P"] = df.apply(lambda row: calculate_gain_contribution(
        row["max_semantic_similarity_precision_bert_rag"],
        row["max_semantic_similarity_precision_bert_no_rag"]), axis=1)

    df["gain_bertscore_R"] = df.apply(lambda row: calculate_gain_contribution(
        row["max_semantic_similarity_recall_bert_rag"],
        row["max_semantic_similarity_recall_bert_no_rag"]), axis=1)

    df["gain_bertscore_f1"] = df.apply(lambda row: calculate_gain_contribution(
        row["max_semantic_similarity_f1_bert_rag"],
        row["max_semantic_similarity_f1_bert_no_rag"]), axis=1)

    df["gain_ce"] = df.apply(lambda row: calculate_gain_contribution(
        row["max_semantic_similarity_ce_rag"],
        row["max_semantic_similarity_ce_no_rag"]), axis=1)

    df["gain_e5"] = df.apply(lambda row: calculate_gain_contribution(
        row["max_semantic_similarity_e5_rag"],
        row["max_semantic_similarity_e5_no_rag"]), axis=1)

    df["gain_dpr"] = df.apply(lambda row: calculate_gain_contribution(
        row["max_semantic_similarity_dpr_rag"],
        row["max_semantic_similarity_dpr_no_rag"]), axis=1)

    df["gain_contriever"] = df.apply(lambda row: calculate_gain_contribution(
        row["max_semantic_similarity_contriever_rag"],
        row["max_semantic_similarity_contriever_no_rag"]), axis=1)

    df["gain_minilm"] = df.apply(lambda row: calculate_gain_contribution(
        row["max_semantic_similarity_minilm_rag"],
        row["max_semantic_similarity_minilm_no_rag"]), axis=1)

    print(f"Semantic similarity analysis completed!")
    gain_columns = [col for col in df.columns if "gain" in col]

    # Optional: Display correlation matrix of gains
    if len(gain_columns) > 1:
        print("\nCorrelation matrix between different gain metrics:")
        plot_corr_map(df, title="\nLlama Model\nE5 ranker", method="pearson")

    1==1