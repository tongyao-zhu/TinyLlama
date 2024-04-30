# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import os
import random
import struct

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}

# def get_fragment_lens(chunk, skip_indices):
#     # adapted from https://github.com/yuzhaouoe/pretraining-data-packing/blob/8ee89732e73e9c5dec5af858289512206a050a0d/packing_dataset.py#L165
#     # need to calculate the fragment lengths for each chunk
#     chunk_size = len(chunk)
#     cur_fragment_lens = []
#     prev = 0
#     for token_idx, token in enumerate(chunk):
#         if token == 2 and token_idx not in skip_indices:
#             cur_fragment_lens.append(token_idx - prev + 1)
#             prev = token_idx + 1
#     if prev != chunk_size:
#         cur_fragment_lens.append(chunk_size - prev)
#     # print("Fragment lens:", cur_fragment_lens)
#     # print("Sum of fragment lens:", sum(cur_fragment_lens))
#     return cur_fragment_lens, len(cur_fragment_lens)

# Optimized function using NumPy
def get_fragment_lens_optimized(chunk, skip_indices):
    skip_indices_set = set(skip_indices)
    is_two = np.where(chunk == 2)[0]
    filtered_indices = np.array([idx for idx in is_two if idx not in skip_indices_set])
    # if len(skip_indices) > 0:
    #     print("Skipper indices:", len(skip_indices), "Filtered indices:", len(filtered_indices))
    # # Adjust how fragment lengths are calculated to match the original function
    if filtered_indices.size > 0:
        fragment_lengths = []
        prev = 0
        for idx in filtered_indices:
            fragment_lengths.append(idx - prev + 1)
            prev = idx + 1
        if prev < len(chunk):
            fragment_lengths.append(len(chunk) - prev)
    else:
        fragment_lengths = [len(chunk)]  # If no valid indices, the entire chunk is one fragment

    return fragment_lengths, len(fragment_lengths)


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

# def calculate_bigram_overlap(prev, next):
#     prev_bigrams = set(zip(prev[:-1], prev[1:]))
#     next_bigrams = set(zip(next[:-1], next[1:]))
#     if len(prev_bigrams) == 0 or len(next_bigrams) == 0:
#         return 0
#     return len(prev_bigrams.intersection(next_bigrams)) / min(len(prev_bigrams), len(next_bigrams))

def calculate_bigram_set_overlap(prev_bigrams, next_bigrams):
    """
    Calculate the overlap between two sets of bigrams
    Args:
        prev_bigrams:
        next_bigrams:

    Returns:

    """
    if len(prev_bigrams) == 0 or len(next_bigrams) == 0:
        return 0
    return len(prev_bigrams.intersection(next_bigrams)) / min(len(prev_bigrams), len(next_bigrams))
def get_eos_indices_between_relevant_docs(chunk, sim_func, eos_id, lower_bound, upper_bound):
    """
    Merge neighboring documents if they have similarity
    Args:
        chunk: an array of tokens
        eos_id: EOS token id used to mark the boundary of documents

    Returns: indices of the EOS tokens that need to be replaced
    """
    eos_indices = np.where(chunk == eos_id)[0]
    docs = [x.tolist() for x in split_into_docs(chunk, eos_id)]
    doc_bigrams = [set(zip(doc[:-1], doc[1:])) for doc in docs]
    # print([len(doc) for doc in doc_bigrams])
    # print("Merging")
    # if two docs are have similarity, replace the eos with the replace_token_id
    result_indices = []
    sim_scores = []
    for i in range(len(docs) - 1):
        sim = sim_func(doc_bigrams[i], doc_bigrams[i+1])
        sim_scores.append(sim)
        if lower_bound <= sim <= upper_bound:
            result_indices.append(eos_indices[i].item())
    # if len(eos_indices) > 0:
    #     print("EOS indices:", eos_indices, "Skip indices:", result_indices, "Percentage {}/{}={}".format(len(result_indices), len(eos_indices), len(result_indices)/len(eos_indices)))
    #     print("Similarity scores:", sim_scores)
    # else:
    #     print("No EOS indices")
    return result_indices

def merge_neighboring_docs(chunk, sim_func, eos_id, replace_token_id, lower_bound, upper_bound):
    """
    Merge neighboring documents if they have similarity
    Args:
        chunk: an array of tokens
        eos_id: EOS token id used to mark the boundary of documents
        replace_token_id: default is 13, which is the '\n' token

    Returns: a new array with the EOS tokens replaced
    """
    to_replace = get_eos_indices_between_relevant_docs(chunk, sim_func, eos_id, lower_bound, upper_bound)
    # assert that the token at each index is the EOS token
    assert all(chunk[i] == eos_id for i in to_replace), "Not all tokens to replace are EOS tokens"
    chunk[to_replace] = replace_token_id
    return chunk

def code(dtype):
    for k in dtypes:
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    def __init__(
        self, filenames, n_chunks, block_size, seed=12345, shuffle=True, wrap=False, num_processes=1, process_rank=0, mask_attn=False, merge_method="none"
    ):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank
        self._mask_attn = mask_attn
        self._merge_method = merge_method

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        return PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
            mask_attn=self._mask_attn,
            merge_method=self._merge_method
        )


class PackedDatasetBuilder(object):
    def __init__(self, outdir, prefix, chunk_size, sep_token, dtype="auto", vocab_size=None):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames = []

    def _write_chunk(self):
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)

        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        self._filenames.append(filename)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr):
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_reminder(self):
        self._write_chunk()


class PackedDatasetIterator:
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap, mask_attn, merge_method):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._wrap = wrap

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None

        self._mmaps = []
        self._buffers = []

        self._block_idxs = []
        self._curr_idx = 0
        print("In iterator, whether we are masking the attention?", mask_attn)
        print("In iterator, the merge method is", merge_method)
        self._mask_attn = mask_attn
        assert self._mask_attn in ["adaptive", "strict", ""], "Mask attn must be either adaptive or strict, but got {}".format(self._mask_attn)
        self._merge_method = merge_method
        if self._mask_attn == "adaptive":
            assert self._merge_method == "overlap", "Merge method must be overlap when mask_attn is adaptive, but got {}".format(self._merge_method)
        self._load_n_chunks()

    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _load_n_chunks(self):
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        if self._n_chunks > len(self._filenames[self._file_idx :]):
            # if not self._wrap:
            #     raise StopIteration
            self._file_idx = 0

        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(filename)
                assert self._chunk_size % self._block_size == 0, "Chunk size {} must be a multiple of block size {}".format(self._chunk_size, self._block_size)
                self._n_blocks = self._chunk_size // self._block_size
            # TODO: check header matches with previous files
            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

        self._file_idx += self._n_chunks
        n_all_blocks = self._n_chunks * self._n_blocks

        self._block_idxs = self._rng.permutation(n_all_blocks) if self._shuffle else range(n_all_blocks)

        self._curr_idx = 0

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
        arr = torch.from_numpy(arr.astype(np.int64)) # block size here is 8193
        # print("Block size", self._block_size, "arr shape", arr.shape, "arr dtype", arr.dtype, "arr", arr)
        self._curr_idx += 1
        if self._merge_method == "overlap" and self._mask_attn == "strict": # only merge neighboring docs when mask_attn is strict
            arr = merge_neighboring_docs(arr, sim_func=calculate_bigram_set_overlap, eos_id=2, replace_token_id=13, lower_bound=0.1, upper_bound=0.5)
        else:
            assert self._merge_method == "no" or (self._merge_method=="overlap" and self._mask_attn=="adaptive") , "Merge method must be either overlap or no, but got {}".format(self._merge_method)
        if self._mask_attn:
            if self._mask_attn == "adaptive":
                skip_eos_indices = get_eos_indices_between_relevant_docs(arr, sim_func=calculate_bigram_set_overlap, eos_id=2, lower_bound=0.1, upper_bound=0.5)
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_optimized(arr[:self._block_size-1], skip_eos_indices)
            else:
                # cur_fragment_lens, cur_fragment_nums = get_fragment_lens(arr[:self._block_size-1], []) # only calculate the input
                cur_fragment_lens, cur_fragment_nums = get_fragment_lens_optimized(arr[:self._block_size-1], [])

                # assert cur_fragment_nums == cur_fragment_nums2, "Fragment nums do not match"
                # assert cur_fragment_lens == cur_fragment_lens2, "Fragment lens do not match"

            # print("Yieleding with mask attn, shapes are : ", arr.shape, len(cur_fragment_lens), cur_fragment_nums)
            return {"idx": arr, "fragment_lens": cur_fragment_lens, "fragment_nums": cur_fragment_nums}
        else:
            return {"idx": arr}

class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)
