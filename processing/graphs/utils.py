import json
import os
import tqdm
from collections import defaultdict
import numpy as np

def get_path_stats(result_paths):
    # print the distribution of lengths in result_paths
    print("Number of paths:", len(result_paths))
    print("Average length:", sum([len(path) for path in result_paths]) / len(result_paths), "standard deviation:", np.std([len(path) for path in result_paths]))
    print("Maximum length:", max([len(path) for path in result_paths]))
    print("Number of paths with length 1:", len([path for path in result_paths if len(path) == 1]))
    print("% of paths with length 1:", len([path for path in result_paths if len(path) == 1]) / len(result_paths)*100, "%")
    print("Top 10 paths length:", sorted([len(path) for path in result_paths], reverse=True)[:10])
    print("Bottom 10 paths length:", sorted([len(path) for path in result_paths])[:10])
    stats = {
        'num_paths': len(result_paths),
        'avg_length': sum([len(path) for path in result_paths]) / len(result_paths),
        'std_length': np.std([len(path) for path in result_paths]),
        'max_length': max([len(path) for path in result_paths]),
        'num_paths_length_1': len([path for path in result_paths if len(path) == 1]),
        'percent_paths_length_1': len([path for path in result_paths if len(path) == 1]) / len(result_paths)
    }
    return stats

def read_jsonl_files(jsonl_dir):
    adj_list = {}
    for file_name in tqdm.tqdm(os.listdir(jsonl_dir)):
        file_path = os.path.join(jsonl_dir, file_name)
        with open(file_path) as f:
            for line in f:
                line = json.loads(line)
                adj_list[line['query_id']] = line['docs']
    print("Read", len(adj_list), "queries from", jsonl_dir)
    print("Read {} files from {}".format(len(os.listdir(jsonl_dir)), jsonl_dir))
    return adj_list

def read_adj_lst(adj_lst_file):
    """
    Supports reading both a single json file and a directory of jsonl files
    Args:
        adj_lst_file:

    Returns: adj_lst

    """
    if adj_lst_file.endswith('.json'):
        # read the adjacency list
        adj_lst = json.load(open(adj_lst_file))
    elif os.path.isdir(adj_lst_file):
        adj_lst = read_jsonl_files(adj_lst_file)
    else:
        raise ValueError("Invalid adj list file")
    return adj_lst

def get_file_ids(directory_path):
    file_ids = set()

    # Iterate over files in the directory
    for filename in tqdm.tqdm(os.listdir(directory_path)):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory_path, filename)

            # Read each JSONL file
            with open(file_path, 'r') as file:
                for line in file:
                    # Parse each line as JSON
                    data = json.loads(line)
                    file_ids.add(data['id'])

    return file_ids

def read_trec_results(file_path):
    results = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()

            query_id = parts[0]
            doc_id = parts[2]
            rank = int(parts[3])
            score = float(parts[4])

            if query_id not in results:
                results[query_id] = []

            results[query_id].append({'doc_id': doc_id, 'score': score, 'rank': rank})

    return results