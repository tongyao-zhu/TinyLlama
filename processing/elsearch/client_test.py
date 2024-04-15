import json
import tqdm
import time
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk, parallel_bulk
from elasticsearch_dsl import MultiSearch, Search

def generate_actions():
    """Reads the file through csv.DictReader() and for each row
    yields a single document. This function is passed into the bulk()
    helper to create many documents in sequence.
    """
    with open(os.path.join(BASE_DIR, f'chunk_{CHUNK_NUM}.jsonl'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            data['_id'] = data['id']
            # client.index(index=f"my_index_{CHUNK_NUM}", id=data['id'], document=data)
            yield data


# taken from the following link:
# https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/getting-started-python.html#_searching_documents
# another refernece for bulk: https://github.com/elastic/elasticsearch-py/blob/main/examples/bulk-ingest/bulk-ingest.py

CHUNK_NUM = 0
BASE_DIR= '/home/aiops/zhuty/ret_pretraining_data/id_added/cc/train/'
filename = os.path.join(BASE_DIR, f'chunk_{CHUNK_NUM}.jsonl')
# get the number of lines in the file
num_lines = sum(1 for line in open(filename))
print("Number of lines in the file: ", num_lines)


ELASTIC_PASSWORD="YL8RhK0Ua*PXt_Fn1LFW"

client = Elasticsearch(
    "https://localhost:9201",  # Elasticsearch endpoint
    ca_certs='/home/aiops/zhuty/elasticsearch-8.12.1/config/certs/http_ca.crt',
    basic_auth=('elastic', ELASTIC_PASSWORD),  # HTTP basic authentication),
    # api_key=('api-key-id', 'api-key-secret'),  # API key ID and secret
)

client.info()


client.indices.delete(index=f"my_index_{CHUNK_NUM}")

client.indices.create(index=f"my_index_{CHUNK_NUM}")


# read the jsonl file
print("Start adding to index...")
start_time = time.time()
success = 0
progress = tqdm.tqdm(unit = "docs", total = num_lines)
for ok, action in parallel_bulk(client = client, index = f"my_index_{CHUNK_NUM}", actions = generate_actions(),
                                 chunk_size=1000, thread_count=32):
    progress.update(1)
    success += ok


# with open(os.path.join(BASE_DIR, f'chunk_{CHUNK_NUM}.jsonl'), 'r') as f:
#     lines = f.readlines()
#     for line in tqdm.tqdm(lines, desc = "add to index"):
#         data = json.loads(line)
#         client.index(index=f"my_index_{CHUNK_NUM}", id=data['id'], document=data)
end_time = time.time()
print("Indexed %d/%d documents" % (success, num_lines))
print("Finish adding to index, time: ", end_time - start_time)

file = client.get(index=f"my_index_{CHUNK_NUM}", id='0_0')
print(file)
# print(client.in)

queries = os.path.join("/home/aiops/zhuty/ret_pretraining_data/id_added/cc/queries/first", 'chunk_0.jsonl')
loaded_queries = [json.loads(line) for line in open(queries, 'r').readlines()]
print("Loaded queries in total: ", len(loaded_queries))

def process_results(results):
    top_results = []
    for hits in results['hits']['hits']:
        # print( hits['_id'])
        top_results.append(hits['_id'])
    return top_results

outputs = []

# loaded_queries = [{'id': query['id'], 'title': query['title'],
#                    'match': {"contents": query['title']}} for query in loaded_queries]
BATCH_SIZE=100

splitted_queries = [loaded_queries[i:i+BATCH_SIZE] for i in range(0, len(loaded_queries), BATCH_SIZE)]

# for i , query in enumerate(loaded_queries):
#     s = Search(using=client).query("match", contents=query['match']['contents'])
#     print(s.to_dict())
#     top_docs = process_results(s.execute())
#     outputs.append((query['id'], top_docs))

for i, queries in tqdm.tqdm(enumerate(splitted_queries), desc = "searching", total=len(splitted_queries)):
    ms = MultiSearch(using=client, index=f"my_index_{CHUNK_NUM}")
    for query in queries:
        # print("Adding query: ", query['match']['contents'])
        s = Search().query("match", contents=query['title'])
        ms = ms.add(s)
    result = ms.execute()
    for j,query in enumerate(queries):
        top_docs = process_results(result[j])
        outputs.append((query['id'], top_docs))

    #result = client.msearch(searches = queries, index = f"my_index_{CHUNK_NUM}", max_concurrent_searches=96)
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
file = open(os.path.join("/home/aiops/zhuty/ret_pretraining_data/id_added/cc/elsearch_results/first", f'chunk_{CHUNK_NUM}_result.jsonl'), 'w')
for output in outputs:
    file.write(json.dumps(output) + "\n")

# print(result)