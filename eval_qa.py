import ast
import re
import sys
import time
import pickle
import string
import argparse
import traceback
import unicodedata

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from ralm.file_utils import print_args
from telegram_logs import TelegramHandler
from ralm.model_utils import load_model_and_tokenizer
from ralm.retrievers.retrieval_factory import add_retriever_args, get_retriever


TELEGRAM_HANDLER = TelegramHandler()
RETRIEVAL_TYPES = [
    "manual-dense-faiss",
    "dense",
    "sparse",
]
PROMPT_PHRASE = "Answer the question concisely"


def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]


def build_qa_prompt(example, num_docs=1):
    q = normalize_question(example["question"])

    if num_docs == 0:
        return [
            {"role": "system", "content": "You are an AI assistant tasked with answering questions concisely."},
            {"role": "user", "content": f"Answer the question concisely.\n\n# Question:\n{q}\n\n# Answer:"}
        ]

    retrieved_docs = example.get("retrieved_docs", [])[:num_docs]
    docs_text = "\n\n".join([f"# Passage {i+1}:\n{doc['text']}" for i, doc in enumerate(retrieved_docs)])

    return [
        {"role": "system", "content": "You are an AI assistant tasked with answering questions concisely."},
        {"role": "user", "content": f"Answer the question concisely, based on the following passages.\n\n{docs_text}\n\n# Question:\n{q}\n\n# Answer:"}
    ]


def normalize_answer(s):
    """Normalize answer."""
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return str(text).lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def text_has_answer(answers, text) -> bool:
    if isinstance(answers, str):
        answers = [answers]
    text = normalize_answer(text)
    for single_answer in answers:
        single_answer = normalize_answer(single_answer)
        if single_answer in text:
            return True
    return False


def exact_match(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def evaluate_dataset(
        model, tokenizer, device, eval_dataset, max_length, num_docs=0, max_tokens_to_generate=300
):
    from math import ceil

    TELEGRAM_HANDLER.emit("Start evaluating QA model with batching...")
    idx = 0
    num_correct = 0
    num_has_answer = 0
    total_samples = len(eval_dataset)
    next_threshold = 10
    batch_size = get_batch_size(num_docs)
    words_count = 0
    results = []

    def chunks(dataset, batch_size):
        dataset_len = len(dataset)
        for i in range(0, dataset_len, batch_size):
            yield dataset.select(range(i, min(i + batch_size, dataset_len)))


    for batch in (tq := tqdm(chunks(eval_dataset, batch_size), total=ceil(total_samples / batch_size), desc="EM: 0.0%")):
        prompts = [tokenizer.apply_chat_template(build_qa_prompt(ex, num_docs=num_docs), add_generation_prompt=True, tokenize=False) for ex in batch]
        answers_list = [ex["answers"] for ex in batch]
        has_answers = [text_has_answer(ans, prompt.split(PROMPT_PHRASE)[-1]) for ans, prompt in zip(answers_list, prompts)]

        # Tokenize batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.module.generate(
                **inputs,
                max_new_tokens=max_tokens_to_generate,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generations = get_answer_from_model_output(outputs, tokenizer)

        for i in range(len(batch)):
            prediction = generations[i]
            answers = answers_list[i]
            has_answer = has_answers[i]

            words = re.findall(r'\b\w+\b', prediction)
            words_count += len(words)

            is_correct = any([exact_match(prediction, answer) for answer in answers])
            idx += 1
            if is_correct:
                num_correct += 1
            if has_answer:
                num_has_answer += 1
            results.append({#"question_id": batch[i]["question_id"],
                            "question": batch[i]["question"],
                            "answer": answers,
                            "has_answer": has_answer,
                            "prediction": prediction,
                            "EM": int(is_correct)})

        current_em = (num_correct / idx) * 100
        tq.set_description(f"EM: {current_em:4.1f}%")
        progress = (idx / total_samples) * 100
        if progress >= next_threshold:
            TELEGRAM_HANDLER.emit(f"""<b>Progress:</b> {int(progress)}% |<b> EM:</b> {current_em:4.1f}% <b>| AVG words in output:</b> {words_count/idx:.1f}""")
            next_threshold += 10

        # Free GPU memory
        import gc
        del inputs["attention_mask"], inputs["input_ids"], inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print(f"EM: {current_em:.1f}%")
    print(f"% of prompts with answer: {num_has_answer / idx * 100:.1f}%")
    print(f"AVG words in output: {words_count/idx:.1f}")
    TELEGRAM_HANDLER.emit(f"""Finished evaluating QA model
<b>EM:</b> {current_em:.1f}%
<b>% of prompts with answer:</b> {num_has_answer / idx * 100:.1f}%""")

    results_df = pd.DataFrame(results)
    1 == 1



def get_answer_from_model_output(outputs, tokenizer):
    generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [g.split("# Answer:")[-1] for g in generations]


def get_batch_size(num_docs):
    if num_docs <= 10:
        return 32
    elif num_docs <= 25:
        return 12
    elif num_docs <= 50:
        return 4
    else:
        return 1


def load_data(args):
    print("Loading dataset:", args.dataset_path)
    TELEGRAM_HANDLER.emit(f"<b>Loading QA Dataset:</b> {args.dataset_path}")

    # Load dataset
    if args.load_from == "hf":
        dataset = load_dataset(
            args.dataset_path,
            args.dataset_name,
            split=args.dataset_split
        )
    else:
        dataset = load_dataset(
            'json',
            data_files=args.dataset_path
        )['train']

    # Shuffle and sample
    dataset = dataset.shuffle(seed=1971).select(range(args.sample_size))

    # Normalize answers in-place to hf datasets
    if "mandarjoshi/trivia_qa" in args.dataset_path:
        dataset = dataset.map(lambda d: {
            **d,
            "answers": d["answer"]["aliases"] + d["answer"]["normalized_aliases"]
        })
    elif "google-research-datasets/nq_open" in args.dataset_path:
        dataset = dataset.map(lambda d: {
            **d,
            "answers": d["answer"]
        })
    elif "akariasai/PopQA" in args.dataset_path:
        dataset = dataset.map(lambda d: {
            **d,
            "answers": ast.literal_eval(d["possible_answers"])
        })

    return dataset



def get_cache(args):
    if args.cache_dir is not None:
        # Only keep relevant keys
        important_keys = [
            "sample_size", "dataset_name", "dataset_path", "dataset_split",
            "embedder_name", "index_name", "load_from", "num_docs",
            "retrieval_type", "ranking_strategy"
        ]

        config_string = "\n".join(f"{key}={val}" for key, val in sorted(vars(args).items()) if key in important_keys)

        with open(args.cache_dir, 'rb') as f:
            cache_data = pickle.load(f)

        return cache_data[config_string]


def main(args):
    print_args(args, output_dir=args.output_dir)
    assert args.num_docs <= args.num_retrival_docs, "num_retrival_docs must be grater than num_docs."

    print("Loading model:", args.model_name)
    TELEGRAM_HANDLER.emit(f"<b>Loading model:</b> {args.model_name}")
    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name, model_parallelism=args.model_parallelism, auth_token=args.auth_token#, temperature=0.3, top_p=0.8,#cache_dir=args.cache_dir
    )
    model_max_length = config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings

    eval_dataset = load_data(args)

    if args.num_docs > 0:
        print(f"Creating retriever of type {args.retrieval_type}...")
        TELEGRAM_HANDLER.emit(f"Creating retriever of type {args.retrieval_type}...")
        TELEGRAM_HANDLER.emit(f"Loading {args.index_name} index\nand {args.embedder_name} embedder")
        retriever = get_retriever(args.retrieval_type, args, tokenizer)
        print(f"Retrieving {args.num_docs} documents for each query...")
        TELEGRAM_HANDLER.emit(f"Retrieving {args.num_docs} documents for each query...")
        start_time = time.time()
        eval_dataset = retriever.retrieve(eval_dataset, k=args.num_retrival_docs)
        print(f"Retrieving Time: {(time.time() - start_time)/60:.2f}m")


    evaluate_dataset(
        model, tokenizer, device, eval_dataset,
        max_length=model_max_length,
        num_docs=args.num_docs,
    )


if __name__ == '__main__':
    assert sys.argv[1] == "--retrieval_type"
    retrieval_type = sys.argv[2]

    assert retrieval_type in RETRIEVAL_TYPES

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    # Model params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_docs", type=int, default=0)

    # Dataset params
    parser.add_argument("--load_from", type=str, choices=["hf", "file"], default="hf")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="validation")
    parser.add_argument("--sample_size", type=int, default=500)

    # retrieval params
    parser.add_argument("--ranking_strategy", type=str, choices=["first", "logprob", "oracle", "random", "rerank"], default="first")
    parser.add_argument("--num_retrival_docs", type=int, default=0)
    parser.add_argument("--retrieval_type", required=True, choices=RETRIEVAL_TYPES)
    parser.add_argument("--index_name", type=str, default="create_index/downloads/data/wikipedia_split/psgs_w100_contriever.index")
    parser.add_argument("--docid_mapping", type=str, default="create_index/downloads/data/wikipedia_split/psgs_w100_metadata_contriever.jsonl")
    add_retriever_args(parser, retrieval_type)

    args = parser.parse_args()

    try:
        TELEGRAM_HANDLER.emit(f"""###########################
<b>Model Name:</b> {args.model_name.split('/')[-1]}
<b>QA Dataset:</b> {args.dataset_path.split('/')[-1]}
<b>Data #queries:</b> {args.sample_size:.0f}
<b>Num Retrieval Docs:</b> {args.num_retrival_docs:.0f}
<b>Num Docs:</b> {args.num_docs}
<b>Retrieval Type:</b> {args.retrieval_type if args.num_docs > 0 else "Not Retrieval"}
<b>Embedder Name:</b> {args.embedder_name if args.num_docs > 0 else "Not Retrieval"}
<b>Index Name:</b> {args.index_name if args.num_docs > 0 else "Not Retrieval"}
<b>DocID Mapping:</b> {args.docid_mapping if args.num_docs > 0 else "Not Retrieval"}
""")

        #############
        # Main      #
        #############
        main(args)
        TELEGRAM_HANDLER.close()
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        TELEGRAM_HANDLER.emit(error_message)
        print(error_message)
        TELEGRAM_HANDLER.close()