import csv
import json
from tqdm import tqdm


def tsv_to_jsonl(tsv_path, jsonl_path):
    with open(tsv_path, 'r', encoding='utf-8') as tsv_file, \
         open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        for row in tqdm(reader, total=21015324, desc="Converting TSV to JSONL"):
            json_line = json.dumps({
                'id': row['id'],
                'title': row['title'],
                'contents': row['text']
            }, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')

# Example usage:
tsv_to_jsonl('/lv_local/home/or.dado/PycharmProjects/RAG/create_index/downloads/data/wikipedia_split/psgs_w100.tsv',
             '/create_index/downloads/data/wikipedia_split/w100_luence_index/psgs_w100.jsonl')


# python -m pyserini.index.lucene \
#   --collection JsonCollection \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 28 \
#   --input /lv_local/home/or.dado/PycharmProjects/RAG/create_index/downloads/data/wikipedia_split/w100_luence_index \
#   --index /lv_local/home/or.dado/PycharmProjects/RAG/create_index/downloads/data/wikipedia_split/w100_luence_index \
#   --storePositions --storeDocvectors --storeRaw