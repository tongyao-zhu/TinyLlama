import tqdm
import json
import argparse
from utils import read_trec_results
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--search_type", required=True)
    parser.add_argument("--top_k", required=True, type=int)
    parser.add_argument("--chunk_num_lower", required=True, type=int)
    parser.add_argument("--chunk_num_upper", required=True, type=int)
    parser.add_argument("--low_score_threshold", required=False, default=20, type=float)
    return parser.parse_args()


args = parse_args()

version = args.version
search_type = args.search_type

result_dir = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{version}/{search_type}_search_results/"

result_dict = {}
for chunk_num in tqdm.tqdm(range(args.chunk_num_lower, args.chunk_num_upper + 1)):
    result_dict.update(read_trec_results(os.path.join(result_dir, f"chunk_{chunk_num}.result.txt")))


top_k = args.top_k
# add the top_k neighbors as edges
low_score_threshold = args.low_score_threshold
if search_type == 'dense' and args.low_score_threshold == 20:
    low_score_threshold = 0 # set the threshold to 0 for dense search

adj_lst = {}
num_edges = 0
for query_id, docs in tqdm.tqdm(result_dict.items()):
    adj_lst[query_id] = []
    for doc in docs[:top_k + 1]:
        if doc['doc_id'] == query_id:
            # neighbor is the query itself, continue
            continue
        if doc['score'] < low_score_threshold:
            # filter out the low score neighbors
            continue
        adj_lst[query_id].append((doc['doc_id'], doc['score']))
        num_edges += 1
print("Number of edges:", num_edges)
# get the average out degree
print("Average out degree:", num_edges / len(result_dict))
print("max out degree:", max([len(neighbors) for neighbors in adj_lst.values()]))
print("std out degree:", np.std([len(neighbors) for neighbors in adj_lst.values()]))
# get the average in degree
# in_degree = {}
# for query_id, neighbors in tqdm.tqdm(adj_lst.items()):
#     for neighbor in neighbors:
#         if neighbor[0] not in in_degree:
#             in_degree[neighbor[0]] = 0
#         in_degree[neighbor[0]] += 1
# print("Average in degree:", sum(in_degree.values()) / len(result_dict))
# print("max in degree:", max(in_degree.values()))
# print("min in degree:", min(in_degree.values()))
# print("std in degree:", np.std(list(in_degree.values())))

json.dump(adj_lst, open(
    f"/home/aiops/zhuty/ret_pretraining_data/id_added/{version}/adj_lists/{search_type}/chunked/adj_lst_top_{top_k}_chunk_{args.chunk_num_lower}-{args.chunk_num_upper}.json",
    "w"))
