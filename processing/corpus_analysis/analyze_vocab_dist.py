# SEt the working dire
import os
from pathlib import Path
import numpy as np
import tqdm
# os.chdir('/home/aiops/zhuty/tinyllama')
# add to PYTHONPATH
import sys
sys.path.append('/home/aiops/zhuty/tinyllama')
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt import Tokenizer
import json

ds_name = sys.argv[1]

# BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/sample_processed/{ds_name}/"
BASE_DIR=f"/nfs-share/multilingual_synthetic_dataset/madlad/{ds_name}/syn"
if ds_name == "en":
    BASE_DIR="/nfs-share/multilingual_synthetic_dataset/cosmopedia_en_dataset/english_bins"

assert os.path.exists(BASE_DIR), f"Path {BASE_DIR} does not exist"

all_filenames = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.startswith("train")]
print("Number of files:", len(all_filenames))
filenames = all_filenames

split = "train"
dataset = PackedDataset(
    filenames,
    # n_chunks control the buffer size.
    # Note that the buffer size also impacts the random shuffle
    # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
    n_chunks=100 if split == "train" else 2,
    block_size=2048 + 1,
    shuffle=False,
    seed=42,
    num_processes=10,
    process_rank=0,
    merge_method= "no",
    mask_attn=""
)



upper_limit = 1000000
# load llama tokenizer
tokenizer_path = Path('/home/aiops/zhuty/tinyllama/models')
tokenizer = Tokenizer(tokenizer_path)
print("Tokenizer EOS token:", tokenizer.eos_id, "Token is:", tokenizer.decode(np.array([tokenizer.eos_id])) )
print("Tokenizer BOS token:", tokenizer.bos_id)
print("token id 13 token is ", tokenizer.decode(np.array([13])))

EOS_TOKEN = tokenizer.eos_id
print("EOS token:", EOS_TOKEN)

def update_vocab_count(vocab_dict, np_array):
    for item in np_array:
        item = item.item()
        if item in vocab_dict:
            vocab_dict[item] += 1
        else:
            vocab_dict[item] = 1

vocab_count = {}
for i, item in tqdm.tqdm(enumerate(dataset), desc = "Processing"):
    # print("Item keys", item.keys())
    item = item['idx']
    if i >= upper_limit:
        break

    # Find the indices where EOS token occurs
    # print("EOS indices:", eos_indices)
    update_vocab_count(vocab_count, item)


# Save the vocab count
json.dump(vocab_count, open(f"vocab_count_{ds_name}.json", "w"))