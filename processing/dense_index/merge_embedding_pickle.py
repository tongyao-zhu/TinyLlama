import pandas as pd
import json
import time
import tqdm
import sys
import os

# Path to your JSON Lines file


dfs = []

for shard_num in range(0, 8):
    print("READING SHARD", shard_num)
    file_path = f"/home/aiops/zhuty/ret_pretraining_data/id_added/cc/dense_queries/shard_{shard_num}_embedding.pkl"
    df = pd.read_pickle(file_path)
    dfs.append(df)

all_df = pd.concat(dfs)
print("Writing to pickle file")
all_df.to_pickle("/home/aiops/zhuty/ret_pretraining_data/id_added/cc/dense_queries/all_embedding.pkl")

