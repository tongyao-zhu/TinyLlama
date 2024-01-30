import subprocess
import sys

import json
import os

dataset_name = sys.argv[1]

result_dict = {}
result_dir = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{dataset_name}/adj_lists/bm25/chunked/"
for files in os.listdir(result_dir):
    new_dict = json.load(open(os.path.join(result_dir, files)))
    assert len(set(new_dict.keys()).intersection(set(result_dict.keys()))) == 0
    result_dict.update(new_dict)
print("Length of result dict:", len(result_dict))
json.dump(result_dict, open(f"/home/aiops/zhuty/ret_pretraining_data/id_added/{dataset_name}/adj_lists/bm25/adj_lst_top_100.json", "w"))