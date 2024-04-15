import json
import tqdm
import time
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk, parallel_bulk
from elasticsearch_dsl import MultiSearch, Search
ELASTIC_PASSWORD="YL8RhK0Ua*PXt_Fn1LFW"

port = 9201

client = Elasticsearch(
    f"https://localhost:{port}",  # Elasticsearch endpoint
    ca_certs='/home/aiops/zhuty/elasticsearch-8.12.1-1/config/certs/http_ca.crt',
    basic_auth=('elastic', ELASTIC_PASSWORD),  # HTTP basic authentication),
    # api_key=('api-key-id', 'api-key-secret'),  # API key ID and secret
)

client.info()
# get the size of the index
# Retrieve all indices using the cat API
indices = client.cat.indices(index="*", h="index", s="index:asc")  # 'h' specifies the header of the column we want (index names), 's' specifies sorting

# Split the result into a list (each line is an index name)
index_names = indices.strip().split('\n')
for name in index_names:

    print(name)
    print(client.count(index=name)['count'])
    print("Number of documents in the index: ", client.count(index=name)['count'])
    if name != 'cc_index_first':
        # delete
        client.indices.delete(index=name)
