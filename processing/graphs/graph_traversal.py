import argparse
import multiprocessing
from typing import List, Dict
import json
import time
import datetime
import tqdm
from graph import Graph, Node
from utils import read_trec_results, get_file_ids
import os
import numpy as np
# write the traversal algorithm
def max_tsp_traverse(graph: Graph):
    traversal_paths = []
    curr_path = []
    visited = set()
    num_visited = 0
    while graph.get_node_count() > 0:
        d_i = graph.get_min_degree_node()
        # print("Exploring node", d_i.get_doc_id())

        curr_path.append(d_i)
        # print(graph.get_node_neighbors(d_i.get_doc_id()), graph.get_node_ids())
        best_available_neighbor = graph.get_best_available_neighbor(d_i.get_doc_id())
        # print("best available neighbor", best_available_neighbor)
        # visited.add(d_i.get_doc_id())
        num_visited += 1
        # print("Starting from", d_i.get_doc_id(), "visited", num_visited , "nodes")
        graph.delete_node(d_i.get_doc_id())
        while best_available_neighbor:
            d_j = best_available_neighbor
            # print("Selected neighbor", d_j.get_doc_id())
            d_i = d_j
            curr_path.append(d_i)
            best_available_neighbor = graph.get_best_available_neighbor(d_i.get_doc_id())
            # visited.add(d_i.get_doc_id())
            num_visited += 1
            graph.delete_node(d_i.get_doc_id())

        # finished one round of dfs
        traversal_paths.append(curr_path)
        curr_path = []
    return traversal_paths

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default='/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/train/')
    # parser.add_argument('--result_dir', type=str, default='/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/bm25_search_results/')
    parser.add_argument('--adj_list_file', type=str, default='/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/adj_lists/')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()

# def read_search_results():
#     # Create a Manager to share data between processes
#     manager = multiprocessing.Manager()
#     result_list = manager.list()
#
#     # Create a pool of processes
#     with multiprocessing.Pool() as pool:
#         # Use tqdm to display progress
#         jobs = []
#         for i in tqdm.tqdm(list(range(0, 89)) + ['search_fail_queries_added']):
#             file_path = '/home/aiops/zhuty/ret_pretraining_data/redpajama_2b_id_added/bm25_search_results/chunk_{i}.result.txt'.format(
#                 i=i)
#
#             # Use pool.apply_async to run read_trec_results in a separate process
#             job = pool.apply_async(read_trec_results, (file_path,))
#             jobs.append(job)
#
#         # Wait for all processes to complete and get their results
#         for job in jobs:
#             result_list.append(job.get())
#
#     # Merge individual dictionaries into a final result dictionary
#     final_result_dict = {}
#     for result_dict in result_list:
#         final_result_dict.update(result_dict)
    # return final_result_dict

# def read_and_update(file_path, shared_dict):
#     # Your read_trec_results function will be called here
#     result = read_trec_results(file_path)
#     shared_dict.update(result)

def main(args):

    # get all file ids
    train_file_path = args.train_data_dir
    all_file_ids = get_file_ids(train_file_path)
    if args.test:
        all_file_ids = list(all_file_ids)[:1000]
    # Add the documents as nodes
    graph = Graph()
    for doc_id in all_file_ids:
        graph.add_node(Node(doc_id))

    print("Number of nodes:", graph.get_node_count())

    # TODO: migrate this function out of graph_traversal.py
    # read the adjacency list
    adj_list = json.load(open(args.adj_list_file))
    for query_id, doc_score_lst in adj_list.items():
        for doc_id, score in doc_score_lst:
            graph.add_edge(graph.get_node(query_id), graph.get_node(doc_id), score, directed=True)

    print("Number of edges:", len(graph.get_edges()))

    graph.build_min_heap()
    print("Built min heap", len(graph.min_heap))
    # traverse the graph
    start_time = time.time()
    print("Start traversing the graph...")
    result_paths = max_tsp_traverse(graph)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    # print the distribution of lengths in result_paths
    print("Number of paths:", len(result_paths))
    print("Average length:", sum([len(path) for path in result_paths]) / len(result_paths), "standard deviation:", np.std([len(path) for path in result_paths]))
    print("Maximum length:", max([len(path) for path in result_paths]))

    # convert the elements in nested path into doc ids
    result_path = []
    for path in result_paths:
        result_path.append([node.get_doc_id() for node in path])
    return result_path

if __name__ == '__main__':
    args = parse_args()
    result_path = main(args)
    formatted_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_name = args.adj_list_file.split('/')[-1].split('.')[0]
    result_file_name = os.path.join(os.path.dirname(os.path.dirname(args.adj_list_file)),
                                    "traversal_paths", f'result_path_{path_name}_{formatted_time}.json')
    print("Saving result to", result_file_name)
    json.dump(result_path, open(result_file_name, 'w'))