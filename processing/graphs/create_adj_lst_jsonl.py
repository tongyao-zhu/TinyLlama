import tqdm
import json
import argparse
from utils import read_trec_results
import numpy as np
import os

# a more memory efficient version for getting the adj lists
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--search_type", required=True)
    parser.add_argument("--query_type", required=True)
    parser.add_argument("--chunk_num", required=True, type=int)
    return parser.parse_args()


args = parse_args()

version = args.version
search_type = args.search_type
query_type = args.query_type

result_dir = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{version}/{search_type}_search_results/{query_type}"

result_dict = read_trec_results(os.path.join(result_dir, f"chunk_{args.chunk_num}.result.txt"))

# write the result_list to a jsonl file
with open(os.path.join(result_dir, f"result_{args.chunk_num}.jsonl"), 'w') as outfile:
    for query_id, docs in tqdm.tqdm(result_dict.items()):
        doc_score_lst = [(doc['doc_id'], doc['score']) for doc in docs]
        outfile.write(json.dumps({"query_id": query_id, "docs": doc_score_lst}) + '\n')