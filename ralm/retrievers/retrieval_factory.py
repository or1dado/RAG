def add_retriever_args(parser, retriever_type):
    if retriever_type == "sparse":
        parser.add_argument("--index_name", type=str, default="wikipedia-dpr")
        parser.add_argument("--num_tokens_for_query", type=int, default=32)
        parser.add_argument("--forbidden_titles_path", type=str, default="ralm/retrievers/wikitext103_forbidden_titles.txt")

    elif retriever_type in ["dense", "manual-dense-faiss"]:
        parser.add_argument("--embedder_name", type=str, default="dpr", choices=["intfloat/e5-large-v2","facebook/contriever","facebook/dpr-question_encoder-multiset-base", "dpr", "bert", "spider"])
    else:
        raise ValueError


def get_retriever(retriever_type, args, tokenizer):
    if retriever_type == "sparse":
        from ralm.retrievers.sparse_retrieval import SparseRetriever
        return SparseRetriever(
            tokenizer=tokenizer,
            index_name=args.index_name,
            forbidden_titles_path=args.forbidden_titles_path,
            num_tokens_for_query=args.num_tokens_for_query,
        )
    elif retriever_type == "dense":
        from ralm.retrievers.dense_retrieval import DenseRetriever
        return DenseRetriever(
            tokenizer=tokenizer,
            index_name=args.index_name,
            args=args,
        )
    elif retriever_type == "manual-dense-faiss":
        from ralm.retrievers.manual_dense_retrieval_faiss import DenseRetriever
        return DenseRetriever(
            tokenizer=tokenizer,
            index_name=args.index_name,
            args=args,
        )
    raise ValueError
