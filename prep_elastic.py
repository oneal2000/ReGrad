import argparse
import glob
import time
import json
import csv
from tqdm import tqdm
from src.retrieve.beir.beir.retrieval.search.lexical.elastic_search import ElasticSearch

def build_elasticsearch(
    beir_corpus_file_pattern: str,
    index_name: str,
):
    beir_corpus_files = glob.glob(beir_corpus_file_pattern)
    print(f'#files {len(beir_corpus_files)}')
    config = {
        'hostname': 'http://localhost:9200',
        'index_name': index_name,
        'keys': {'title': 'title', 'body': 'txt'},
        'timeout': 100,
        'retry_on_timeout': True,
        'maxsize': 24,
        'number_of_shards': 'default',
        'language': 'english',
    }
    es = ElasticSearch(config)

    print(f'create index {index_name}')
    es.delete_index()
    time.sleep(5)
    es.create_index()

    def generate_actions():
        for beir_corpus_file in beir_corpus_files:
            # 处理.jsonl文件
            if beir_corpus_file.endswith(".jsonl"):
                with open(beir_corpus_file, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        doc = json.loads(line.strip())
                        _id = doc.get('_id', None)
                        text = doc.get('text', '')
                        title = doc.get('title', '')
                        es_doc = {
                            '_id': _id,
                            '_op_type': 'index',
                            'refresh': 'wait_for',
                            config['keys']['title']: title,
                            config['keys']['body']: text,
                        }
                        yield es_doc
            # 其他类型文件，如.tsv
            else:  
                with open(beir_corpus_file, 'r', encoding='utf-8') as fin:
                    reader = csv.reader(fin, delimiter='\t')
                    header = next(reader)  
                    for row in reader:
                        if len(row) < 3:
                            continue
                        _id, text, title = row[0], row[1], row[2]
                        es_doc = {
                            '_id': _id,
                            '_op_type': 'index',
                            'refresh': 'wait_for',
                            config['keys']['title']: title,
                            config['keys']['body']: text,
                        }
                        yield es_doc

    progress = tqdm(unit='docs')
    es.bulk_add_to_index(
        generate_actions=generate_actions(),
        progress=progress
    )
    print(f"index '{index_name}' has been successfully built.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='input file pattern')
    parser.add_argument("--index_name", type=str, default=None, help="index name")
    args = parser.parse_args()
    build_elasticsearch(args.data_path, index_name=args.index_name)
