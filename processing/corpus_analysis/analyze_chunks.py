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


ds_name = sys.argv[1]
BASE_DIR = f"/home/aiops/zhuty/ret_pretraining_data/sample_processed/{ds_name}/"

all_filenames = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.startswith("train_cc_")]
print("Number of files:", len(all_filenames))
filenames = all_filenames[:10000]

split = "train"
dataset = PackedDataset(
    filenames,
    # n_chunks control the buffer size.
    # Note that the buffer size also impacts the random shuffle
    # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
    n_chunks=512 if split == "train" else 2,
    block_size=8192 + 1,
    shuffle=False,
    seed=42,
    num_processes=10,
    process_rank=0,
    merge_method= "no",
    mask_attn=""
)

upper_limit = 100000
# load llama tokenizer
tokenizer_path = Path('/home/aiops/zhuty/tinyllama/models')
tokenizer = Tokenizer(tokenizer_path)
print("Tokenizer EOS token:", tokenizer.eos_id, "Token is:", tokenizer.decode(np.array([tokenizer.eos_id])) )
print("Tokenizer BOS token:", tokenizer.bos_id)
print("token id 13 token is ", tokenizer.decode(np.array([13])))

EOS_TOKEN = tokenizer.eos_id
print("EOS token:", EOS_TOKEN)

eos_counts = []
unique_terms_counts = []
unique_bigram_counts = []
not_diverse_count = 0
for i, item in tqdm.tqdm(enumerate(dataset), desc = "Processing"):
    item = item['idx']
    if i >= upper_limit:
        break

    # Find the indices where EOS token occurs
    eos_indices = np.where(item == EOS_TOKEN)[0]
    # print("EOS indices:", eos_indices)
    eos_counts.append(len(eos_indices))


    # # Initialize the start index and the list to hold sub-arrays
    # start_idx = 0
    # sub_arrays = []
    #
    # # Loop through each index and split the array
    # for eos_idx in eos_indices:
    #     sub_arrays.append(item[start_idx:eos_idx])  # Slice array up to the EOS token (exclusive)
    #     start_idx = eos_idx + 1  # Update the start index for the next sub-array
    #
    # # Don't forget to add the last sub-array if the array doesn't end with an EOS token
    # if start_idx < len(item):
    #     sub_arrays.append(item[start_idx:])
    #
    # docs = []
    # # Print the sub-arrays
    # for sub_array in sub_arrays:
    #     docs.append(tokenizer.decode(sub_array))
    #
    # for prev_doc, next_doc in zip(docs[:-1], docs[1:]):
    #     print("prev_doc:", prev_doc[-100:])
    #     print("next_doc:", next_doc[:100])


    # print("first 10 tokens:", item[:10])
    # print("last 10 tokens:", item[-10:])
        # print("EOS token not found in the chunk")
    # get number of  unique terms
    num_unique_terms = len(np.unique(item))
    unique_terms_counts.append(num_unique_terms)

    # Initialize an empty set to hold unique bigrams
    unique_bigrams = set()

    # Iterate through the sequence to create bigrams and add them to the set
    for i in range(len(item) - 1):
        # Create a bigram as a tuple and add it to the set
        unique_bigrams.add((item[i].item(), item[i + 1].item()))
    # print("unique bigrams:", unique_bigrams)

    # The number of unique bigrams is the size of the set
    num_unique_bigrams = len(unique_bigrams)

    unique_bigram_counts.append(num_unique_bigrams)

    if num_unique_terms < 100:
        not_diverse_count += 1
        # print("Chunk", i, "has less than 100 unique terms:", num_unique_terms)
        # print(tokenizer.decode(item)[:10000])

print(f"Percentage of chunks with zero EOS tokens: {eos_counts.count(0)}/{i}={eos_counts.count(0)/i*100:.2f}%")
# get statistics of EOS counts
eos_counts = np.array(eos_counts)
print("Mean EOS counts:", np.mean(eos_counts))
print("Max EOS counts:", np.max(eos_counts))
print("Min EOS counts:", np.min(eos_counts))
print("Std EOS counts:", np.std(eos_counts))




print("No diverse chunks count:", not_diverse_count)
print(f"Percentage of chunks without diverse terms: {not_diverse_count}/{i}={not_diverse_count/i*100:.2f}%")
# get statistics of unique terms of each chunk
unique_terms_counts = np.array(unique_terms_counts)
print("Mean unique terms:", np.mean(unique_terms_counts))
print("Max unique terms:", np.max(unique_terms_counts))
print("Min unique terms:", np.min(unique_terms_counts))
print("Std unique terms:", np.std(unique_terms_counts))

# get statistics of unique bigrams of each chunk
unique_bigram_counts = np.array(unique_bigram_counts)
print("Mean unique bigrams:", np.mean(unique_bigram_counts))
print("Max unique bigrams:", np.max(unique_bigram_counts))
print("Min unique bigrams:", np.min(unique_bigram_counts))
print("Std unique bigrams:", np.std(unique_bigram_counts))


# save the counts to array
np.save(f"{ds_name}_eos_counts.npy", eos_counts)
np.save(f"{ds_name}_unique_terms_counts.npy", unique_terms_counts)
np.save(f"{ds_name}_unique_bigram_counts.npy", unique_bigram_counts)