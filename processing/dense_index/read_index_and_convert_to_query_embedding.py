import pandas as pd
import json
import time
import tqdm
import sys
import os

# Path to your JSON Lines file
shard_num = sys.argv[1]

file_path = f'/s3/ret_pretraining_data/id_added/cc_dense_embeddings/dense_embeddings/shard_{shard_num}/embeddings.jsonl'
# file_path = '/home/aiops/zhuty/ret_pretraining_data/id_added/cc/dense_embeddings/shard_5/embeddings.jsonl'

# Read the file and convert it to a list of dictionaries
data = []
start = time.time()
print("Reading the file")
with open(file_path, 'r') as file:
    for line in tqdm.tqdm(file, disable=True):
        json_dict = json.loads(line)
        data.append(json_dict)
print("Finished reading the file", time.time() - start, "seconds")
# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(data)
embeddings = df
embeddings.to_pickle(os.path.join("/home/aiops/zhuty/ret_pretraining_data/id_added/cc/dense_queries", f'shard_{shard_num}_embedding.pkl'))
# Now 'df' is your pandas DataFrame
