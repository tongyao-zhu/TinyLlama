import pandas as pd
import os

BASE_DIR="/home/aiops/zhuty/ret_pretraining_data/id_added/cc/train"
dfs = []
for i in range(0, 100):
    chunk = pd.read_csv(os.path.join(BASE_DIR, "chunk_{}_lengths.csv".format(i)))
    dfs.append(chunk)
df = pd.concat(dfs, ignore_index=True)

print(df['length'].describe())
print("Total number of tokens (B): ", df['length'].sum()/1e9)

# df = pd.read_csv("chunk_99_lengths.csv")