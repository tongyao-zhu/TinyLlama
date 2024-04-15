import tqdm
import json
import argparse
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

result_dir = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{version}/{search_type}_search_results/{query_type}/adj_lists"
os.makedirs(result_dir, exist_ok=True)
# result_dict = read_trec_results(os.path.join(result_dir, f"chunk_{args.chunk_num}.result.txt"))
result_file = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{version}/{search_type}_search_results/{query_type}/chunk_{args.chunk_num}_result.jsonl"

# read the result_file
result_dict = {}
with open(result_file, 'r') as infile:
    for line in infile:
        result = json.loads(line)
        query_id, doc_score_lst = result
        result_dict[query_id] = doc_score_lst

# write the result_list to a jsonl file
with open(os.path.join(result_dir, f"result_{args.chunk_num}.jsonl"), 'w') as outfile:
    for query_id, doc_score_lst in tqdm.tqdm(result_dict.items()):
        outfile.write(json.dumps({"query_id": query_id, "docs": doc_score_lst}) + '\n')