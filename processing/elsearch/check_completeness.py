import os

for i in range(100):
    path = f"/home/aiops/zhuty/ret_pretraining_data/id_added/cc/elsearch_results/last_120m/chunk_{i}_result.jsonl"
    # print("Exists: ", i, os.path.exists(path))
    if not os.path.exists(path):
        print("Not exist: ", i)
