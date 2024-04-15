import json
import tqdm
import time
import os
import sys
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk, parallel_bulk
from elasticsearch_dsl import MultiSearch, Search

mode = sys.argv[1]

# taken from the following link:
# https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/getting-started-python.html#_searching_documents
# another refernece for bulk: https://github.com/elastic/elasticsearch-py/blob/main/examples/bulk-ingest/bulk-ingest.py

# CHUNK_NUM = 0
if mode == "full":
    BASE_DIR = '/home/aiops/zhuty/ret_pretraining_data/id_added/cc/train/'
    index_name = "my_index_total"
elif mode == "first":
    BASE_DIR = '/home/aiops/zhuty/ret_pretraining_data/id_added/cc/queries/first/'
    index_name = "cc_index_first"

# filename = os.path.join(BASE_DIR, f'chunk_{CHUNK_NUM}.jsonl')
# # get the number of lines in the file
# num_lines = sum(1 for line in open(filename))
# print("Number of lines in the file: ", num_lines)

ELASTIC_PASSWORD = "YL8RhK0Ua*PXt_Fn1LFW"

client = Elasticsearch(
    "https://localhost:9200",  # Elasticsearch endpoint
    ca_certs='/home/aiops/zhuty/elasticsearch-8.12.1/config/certs/http_ca.crt',
    basic_auth=('elastic', ELASTIC_PASSWORD),  # HTTP basic authentication),
    # api_key=('api-key-id', 'api-key-secret'),  # API key ID and secret
)

client.info()

if client.indices.exists(index=index_name):
    client.indices.delete(index=index_name)

client.indices.create(index=index_name)

# read the jsonl file
print("Start adding to index...")
start_time = time.time()


def generate_actions(chunk_num):
    """Reads the file through csv.DictReader() and for each row
    yields a single document. This function is passed into the bulk()
    helper to create many documents in sequence.
    """
    with open(os.path.join(BASE_DIR, f'chunk_{chunk_num}.jsonl'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            data['_id'] = data['id']
            if 'title' in data:
                data['contents'] = data['title']
                del data['title']
            # client.index(index=f"my_index_{CHUNK_NUM}", id=data['id'], document=data)
            yield data


success = 0

for chunk_num in range(0, 100):
    # sleep
    time.sleep(5)
    progress = tqdm.tqdm(unit="docs", total=150000)
    for ok, action in parallel_bulk(client=client, index=index_name, actions=generate_actions(chunk_num),
                                    chunk_size=1000, thread_count=32):
        progress.update(1)
        success += ok

# with open(os.path.join(BASE_DIR, f'chunk_{CHUNK_NUM}.jsonl'), 'r') as f:
#     lines = f.readlines()
#     for line in tqdm.tqdm(lines, desc = "add to index"):
#         data = json.loads(line)
#         client.index(index=f"my_index_{CHUNK_NUM}", id=data['id'], document=data)
end_time = time.time()
print("Indexed %d/%d documents" % (success, 1500000))
print("Finish adding to index, time: ", end_time - start_time)

file = client.get(index=index_name, id='0_0')
print(file)

file = client.get(index=index_name, id='99_0')
print(file)

# print(client.in)
