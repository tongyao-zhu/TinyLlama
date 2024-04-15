import json
import tqdm
import time
import os
import argparse

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk, parallel_bulk
from elasticsearch_dsl import MultiSearch, Search
import sys

# taken from the following link:
# https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/getting-started-python.html#_searching_documents
# another refernece for bulk: https://github.com/elastic/elasticsearch-py/blob/main/examples/bulk-ingest/bulk-ingest.py





def parse_args():
    parser = argparse.ArgumentParser(description="Search the index")
    parser.add_argument("--query_type", type=str, required=True, help="The query type")
    parser.add_argument("--split_num", type=int, default=0, help="split num")
    parser.add_argument("--chunk_num", type=int, required=True, help="The chunk number to search")
    parser.add_argument("--index_name", type=str, required=True, help="The index name to search")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    return parser.parse_args()
ELASTIC_PASSWORD = "YL8RhK0Ua*PXt_Fn1LFW"



args = parse_args()
client = Elasticsearch(
    "https://localhost:9200",  # Elasticsearch endpoint
    ca_certs=f'/home/aiops/zhuty/elasticsearch-8.12.1-{args.split_num}/config/certs/http_ca.crt',
    basic_auth=('elastic', ELASTIC_PASSWORD),  # HTTP basic authentication),
    request_timeout=90,
    # api_key=('api-key-id', 'api-key-secret'),  # API key ID and secret
)

client.info()
client.ping()
# get the size of the index
num_docs = client.count(index=args.index_name)['count']
print("Number of documents in the index: ", num_docs)

if num_docs == 0:
    print("No documents in the index. Exiting...")
    sys.exit(0)




DEBUG = args.debug
CHUNK_NUM = args.chunk_num
QUERY_TYPE=args.query_type
queries = os.path.join(f"/home/aiops/zhuty/ret_pretraining_data/id_added/cc/queries/{QUERY_TYPE}", f'chunk_{CHUNK_NUM}.jsonl')
if not os.path.exists(queries):
    print(f"File {queries} does not exist")
    sys.exit(0)
output_path = os.path.join(f"/home/aiops/zhuty/ret_pretraining_data/id_added/cc/el_search_results/{QUERY_TYPE}",
                         f'chunk_{CHUNK_NUM}_result.jsonl')
if os.path.exists(output_path):
    print(f"File {output_path} already exists. Exiting...")
    sys.exit(0)

loaded_queries = [json.loads(line) for line in open(queries, 'r').readlines()]
if DEBUG:
    loaded_queries = loaded_queries[:100]
print("Loaded queries in total: ", len(loaded_queries))


def process_results(results):
    top_results = []
    for hits in results['hits']['hits']:
        # print( hits['_id'])
        top_results.append((hits['_id'], hits['_score']))
    return top_results


outputs = []

# loaded_queries = [{'id': query['id'], 'title': query['title'],
#                    'match': {"contents": query['title']}} for query in loaded_queries]
BATCH_SIZE = 100

splitted_queries = [loaded_queries[i:i + BATCH_SIZE] for i in range(0, len(loaded_queries), BATCH_SIZE)]

# for i , query in enumerate(loaded_queries):
#     s = Search(using=client).query("match", contents=query['match']['contents'])
#     print(s.to_dict())
#     top_docs = process_results(s.execute())
#     outputs.append((query['id'], top_docs))



for i, queries in tqdm.tqdm(enumerate(splitted_queries), desc="searching", total=len(splitted_queries)):
    ms = MultiSearch(using=client, index=args.index_name)
    for query in queries:
        # print("Adding query: ", query['match']['contents'])
        s = Search().query("match", contents=query['title']).extra(size=100)
        ms = ms.add(s)
    result = ms.execute()
    for j, query in enumerate(queries):
        top_docs = process_results(result[j])
        outputs.append((query['id'], top_docs))

    # result = client.msearch(searches = queries, index = f"my_index_{CHUNK_NUM}", max_concurrent_searches=96)
    # print(len(result))
    # print(result[0].hits)

# for query in tqdm.tqdm(loaded_queries, desc = "searching"):
#     result = client.search(index=f"my_index_{CHUNK_NUM}", query={
#         "match": {
#             "contents": query['title']
#         }
#     })
#     print(len(result['hits']['hits']))
#     top_docs = process_results(result)
#     outputs.append((query['id'], top_docs))

# write the outputs to a file
file = open(output_path, 'w')
for output in outputs:
    file.write(json.dumps(output) + "\n")

# print(result)

# code snippet:
# for i in {10..15}; do
#   tmux new -d -s "r$i" "python search.py --chunk_num $i --index_name my_index_total"
# done