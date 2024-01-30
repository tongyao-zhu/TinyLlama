# each query should get 100 results
import multiprocessing
import os

# change the working directory to the root of the project
os.chdir("/home/aiops/zhuty/tinyllama/processing/graphs")
import tqdm
import argparse
import json
import numpy as np
from utils import read_trec_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--search_type", required=True)
    parser.add_argument("--query_type", required=True)
    return parser.parse_args()


args = parse_args()

version = args.version
search_type = args.search_type
query_type = args.query_type
result_dir = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{version}/{search_type}_search_results/{query_type}"
queries_dir = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{version}/queries/{query_type}"
TEST = False

def read_search_results():
    # Create a Manager to share data between processes
    manager = multiprocessing.Manager()
    result_list = manager.list()

    # Create a pool of processes
    with multiprocessing.Pool() as pool:
        # Use tqdm to display progress
        jobs = []
        for c, chunk_name in tqdm.tqdm(enumerate(query_filenames), total=len(query_filenames),
                                       desc="Reading search results"):
            if TEST and c > 3:
                break
            file_path = os.path.join(result_dir, "{i}.result.txt".format(i=chunk_name.replace(".jsonl", "")))

            # Use pool.apply_async to run read_trec_results in a separate process
            job = pool.apply_async(read_trec_results, (file_path,))
            jobs.append(job)

        # Wait for all processes to complete and get their results
        for job in jobs:
            result_list.append(job.get())

    # Merge individual dictionaries into a final result dictionary
    final_result_dict = {}
    for result_dict in result_list:
        final_result_dict.update(result_dict)
    return final_result_dict

# First, ensure that all queries have been successfully searched, extract chunk
query_filenames = [x.split(".")[0] for x in os.listdir(queries_dir)]
search_result_names = [x.split(".")[0] for x in os.listdir(result_dir)]
if not  set(query_filenames) == set(
        search_result_names):
    print("query filenames and search result filenames are not the same {}".format(
        set(query_filenames) - set(search_result_names)))

result_dict = {}
#result_dict = read_search_results()

for c, chunk_name in tqdm.tqdm(enumerate(sorted(query_filenames)), total=len(query_filenames), desc="Reading search results"):
    if TEST and c > 3:
        break
    file_path = os.path.join(result_dir, "{i}.result.txt".format(i=chunk_name.replace(".jsonl", "")))
    if not os.path.exists(file_path):
        print("file not exists {}".format(file_path))
        continue
    print("Reading search results from {}".format(file_path))
    result_dict.update(read_trec_results(file_path))

print("Passed the assertion test, all queries have been successfully searched")
count = 0
# Print the results
for query_id, docs in result_dict.items():
    print(f"Query {query_id}:")
    count += 1
    for doc in docs[:10]:
        print(f"  Doc ID: {doc['doc_id']}, Score: {doc['score']}, Rank: {doc['rank']}")
    if count > 10:
        break

from collections import Counter

# count how many docs have itself as the top1 neighbor
count = 0
result_lengths = []
problematic_queries = []
for query_id, docs in result_dict.items():
    if docs[0]['doc_id'] == query_id:
        count += 1
    result_lengths.append(len(docs))
    if len(docs) < 100:
        problematic_queries.append(query_id)
print("Number of queries that have itself as the top1 neighbor:", count,
      f"percentage: {count / len(result_dict) * 100:.2f}%")
print(Counter(result_lengths))
print("Number of queries that have less than 100 results:", len(problematic_queries),
      f"percentage: {len(problematic_queries) / len(result_dict) * 100:.2f}%")
problematic_queries[:10]

sequence_set = set()
for i in range(0, 89):
    for j in range(0, 10000):
        sequence_set.add("{i}_{j}".format(i=i, j=j))

missing_queries = []
for query_id in sequence_set:
    if query_id not in result_dict:
        missing_queries.append(query_id)
# missing queries is problematic ==> it means that it cannot even search itself ?

print("Number of queries that are missing:", len(missing_queries),
      f"percentage: {len(missing_queries) / len(sequence_set) * 100:.2f}%")
print("missing queries:", missing_queries[:10])
newly_added_search_fail_queries = set(list(problematic_queries) + missing_queries)
print("Number of newly added search fail queries:", len(newly_added_search_fail_queries))

import json

to_write_data = []
not_searched_query_ids = sorted(missing_queries)
if len(not_searched_query_ids) == 0:
    print("No missing queries, nothing to do")
    exit(0)  # exit normally
for docid in tqdm.tqdm(not_searched_query_ids, total=len(not_searched_query_ids)):
    chunk_id, seq_id = docid.split("_")
    # base_path = f"/home/aiops/zhuty/ret_pretraining_data/redpajama_{version}_id_added/queries"
    base_path = f"/home/aiops/zhuty/ret_pretraining_data/id_added/{version}/queries"
    jsonl_file = os.path.join(base_path, "chunk_{}.jsonl".format(chunk_id))
    with open(jsonl_file, "r") as f:
        # directly go the line
        line = f.readlines()[int(seq_id)]
        data = json.loads(line)
        assert data["id"] == docid

        if len(data['title']) > 1500:
            print("problematic docid", docid)
            print(data['title'])
            print(len(data['title'].split()))
            data['title'] = data['title'][-1500:]
        to_write_data.append(data)

# write to jsonl
with open(os.path.join(base_path, "chunk_missing_queries.jsonl"), "w") as f:
    for data in to_write_data:
        f.write(json.dumps(data) + "\n")

