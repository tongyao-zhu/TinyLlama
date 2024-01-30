import sys
import os
import json
import tqdm
ds_name = sys.argv[1]

all_ids= set()
path = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{ds_name}/train"

# read all jsonl lines in this, extract the id
for file in tqdm.tqdm(os.listdir(path), desc="reading files"):
    if file.endswith(".jsonl"):
        with open(os.path.join(path, file), "r") as f:
            for line in f:
                all_ids.add(json.loads(line)["id"])


# another way of checking correctness of all ids, read the last line of each file


# save all ids to json file
with open(f"/home/aiops/zhuty/ret_pretraining_data/id_added/{ds_name}/all_ids.json", "w") as f:
    json.dump(list(all_ids), f)

