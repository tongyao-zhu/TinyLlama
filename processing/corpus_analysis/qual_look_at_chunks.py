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
filenames = all_filenames

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
)



upper_limit = 200000
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

def calculate_unigram_similarity(list1, list2):
    """
    Compute the Jaccard Similarity between two lists of tokens.
    parameters:
    - list1: List of str, tokens from the first document.
    - list2: List of str, tokens from the second document.
    Returns:
        {'jaccard': float, 'overlap': float, 'dice': float}
    """
    # Convert both lists into sets
    set1 = set(list1)
    set2 = set(list2)
    if len(set1) == 0 or len(set2) == 0:
        return {'jaccard': 0, 'overlap': 0, 'dice': 0}
    return calculate_set_similarity(set1, set2)

def calculate_bigram_similarity(list1, list2):
    """
    Compute the Jaccard Similarity between two lists of tokens.
    parameters:
    - list1: List of str, tokens from the first document.
    - list2: List of str, tokens from the second document.
    Returns:
        {'jaccard': float, 'overlap': float, 'dice': float}
    """
    # Create bigrams from the lists
    bigrams1 = set(zip(list1[:-1], list1[1:]))
    bigrams2 = set(zip(list2[:-1], list2[1:]))
    if len(bigrams1) == 0 or len(bigrams2) == 0:
        return {'jaccard': 0, 'overlap': 0, 'dice': 0}
    return calculate_set_similarity(bigrams1, bigrams2)

def calculate_trigram_similarity(list1, list2):
    """
    Compute the Jaccard Similarity between two lists of tokens.
    parameters:
    - list1: List of str, tokens from the first document.
    - list2: List of str, tokens from the second document.
    Returns:
        {'jaccard': float, 'overlap': float, 'dice': float}
    """
    # Create trigrams from the lists
    trigrams1 = set(zip(list1[:-2], list1[1:-1], list1[2:]))
    trigrams2 = set(zip(list2[:-2], list2[1:-1], list2[2:]))
    if len(trigrams1) == 0 or len(trigrams2) == 0:
        return {'jaccard': 0, 'overlap': 0, 'dice': 0}
    return calculate_set_similarity(trigrams1, trigrams2)

def calculate_set_similarity(set1, set2):
    # Calculate the intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Compute Jaccard Similarity
    jaccard = len(intersection) / len(union)
    overlap = len(intersection) / min(len(set1), len(set2))
    dice = 2 * len(intersection) / (len(set1) + len(set2))
    return {'jaccard': jaccard, 'overlap': overlap, 'dice': dice}

sims_dict = {}
for key in ["unigram", "bigram", "trigram"]:
    for metric in ["jaccard", "overlap", "dice"]:
        sims_dict[f"{key}_{metric}"] = []


def split_into_docs(chunk, eos_id = 2):
    """
    Split the chunk into documents based on the EOS token
    Args:
        chunk:
        eos_id:

    Returns: list of documents
    """
    eos_indices = np.where(chunk == eos_id)[0]
    docs = []
    start_idx = 0
    for eos_idx in eos_indices:
        docs.append(chunk[start_idx:eos_idx])
        start_idx = eos_idx + 1
    if start_idx < len(chunk):
        docs.append(chunk[start_idx:])
    return docs


for i, item in tqdm.tqdm(enumerate(dataset), desc = "Processing"):
    # print("Item keys", item.keys())
    item = item['idx']
    if i >= upper_limit:
        break

    # Find the indices where EOS token occurs
    eos_indices = np.where(item == EOS_TOKEN)[0]
    # print("EOS indices:", eos_indices)
    eos_counts.append(len(eos_indices))

    # Split the chunk into documents
    sub_arrays = split_into_docs(item, eos_id=EOS_TOKEN)
    # print("Each array length:", [len(sub_array) for sub_array in sub_arrays])
    # Print the sub-arrays
    # for sub_array in sub_arrays:
    #     docs.append(tokenizer.decode(sub_array))
    for prev_array, next_array in zip(sub_arrays[:-1], sub_arrays[1:]):
        # assert len(prev_array) > 0 and len(next_array) > 0, "Empty array found for".format(prev_array, next_array)
        prev, next = prev_array.tolist(), next_array.tolist()
        for key in ["unigram", "bigram", "trigram"]:
            for metric in ["jaccard", "overlap", "dice"]:
                similarity = calculate_unigram_similarity(prev, next) if key == "unigram" else \
                    calculate_bigram_similarity(prev, next) if key == "bigram" else \
                        calculate_trigram_similarity(prev, next)
                sims_dict[f"{key}_{metric}"].append(similarity[metric])

    # for prev_doc, next_doc in zip(docs[:-1], docs[1:]):
    #     print("[PREV DOC]:", prev_doc[-1000:])
    #     print("[NEXT DOC]:", next_doc[:100])

for key, scores in sims_dict.items():
    scores = np.array(scores)
    print("Average", key, ":", sum(scores) / len(scores))
    print("Std", key, ":", np.std(scores))
    print("Max", key, ":", max(scores))
    print("Min", key, ":", min(scores))
    np.save(f"{ds_name}_{key}.npy", scores)

# print("Average Jaccard Similarity:", sum(jaccard_sims) / len(jaccard_sims))
# print("Std Jaccard Similarity:", np.std(jaccard_sims))
# print("Max Jaccard Similarity:", max(jaccard_sims))
# print("Min Jaccard Similarity:", min(jaccard_sims))
# print("Average Bigram Jaccard Similarity:", sum(bigram_jaccard_similarity) / len(bigram_jaccard_similarity))
# print("Std Bigram Jaccard Similarity:", np.std(bigram_jaccard_similarity))
# print("Max Bigram Jaccard Similarity:", max(bigram_jaccard_similarity))
# print("Min Bigram Jaccard Similarity:", min(bigram_jaccard_similarity))
# print("Average Trigram Jaccard Similarity:", sum(trigram_jaccard_similarity) / len(trigram_jaccard_similarity))
# print("Std Trigram Jaccard Similarity:", np.std(trigram_jaccard_similarity))
# print("Max Trigram Jaccard Similarity:", max(trigram_jaccard_similarity))
# print("Min Trigram Jaccard Similarity:", min(trigram_jaccard_similarity))
#
# print("Total EOS tokens:", sum(eos_counts))
# # print("Distribution of Jaccard Similarity:", jaccard_sims)
# # save the jaccard similarities
# jaccard_sims = np.array(jaccard_sims)
# np.save(f"{ds_name}_jaccard_sims.npy",jaccard_sims)
# bigram_jaccard_similarity = np.array(bigram_jaccard_similarity)
# np.save(f"{ds_name}_bigram_jaccard_similarity.npy",bigram_jaccard_similarity)
# trigram_jaccard_similarity = np.array(trigram_jaccard_similarity)
# np.save(f"{ds_name}_trigram_jaccard_similarity.npy",trigram_jaccard_similarity)
