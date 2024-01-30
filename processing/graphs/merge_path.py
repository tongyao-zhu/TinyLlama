from utils import get_path_stats, read_jsonl_files, read_adj_lst
import argparse
import json
import os
from tqdm import tqdm
import time
import datetime
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_file", type=str, required=True)
    parser.add_argument("--adj_lst", type=str, required=True)
    parser.add_argument("--cluster_size", type=int, default=21)
    parser.add_argument("--top_k", type=int, default=100)
    return parser.parse_args()


def merge(cluster2docs, doc2cluster, adj_lst, cluster_size=21, top_k=10):
    # data_stats(self.cluster2docs)

    merged_clusters_num = 0
    for cluster, cluster_docs in tqdm(cluster2docs.copy().items()):
        if len(cluster_docs) < cluster_size:
            merged_clusters_num += 1
            # print(merged_clusters_num)
            for doc in cluster_docs:
                # knns, relative_id = self.knns, doc
                # top1k, top1k_cluster = self.output_first_doc_knn_not_in_the_cluster(knns[relative_id, :], cluster)
                to_join_neighbor, to_join_neighbor_cluster = None, None
                if doc in adj_lst:
                    for n, score in adj_lst[doc][:top_k]:
                        # find the first neighbor that is not in the same cluster
                        if doc2cluster[n] == cluster:
                            continue
                        else:
                            to_join_neighbor = n
                            to_join_neighbor_cluster = doc2cluster[to_join_neighbor]
                            break


                # bp()
                k_cluster_docs = cluster2docs[to_join_neighbor_cluster]
                # bp()
                # add k to doc
                # k_cluster_docs.append(k)
                if to_join_neighbor is None:
                    k_cluster_docs.append(doc)
                else:
                    k_cluster_docs.insert(k_cluster_docs.index(to_join_neighbor), doc)

                # update the cluster
                cluster2docs[to_join_neighbor_cluster] = k_cluster_docs
                doc2cluster[doc] = to_join_neighbor_cluster
            del cluster2docs[cluster]
    print(merged_clusters_num)
    return cluster2docs, doc2cluster


def main():
    args = parse_args()
    old_paths = json.load(open(args.path_file))
    get_path_stats(old_paths)
    adj_lst = read_adj_lst(args.adj_lst)
    cluster2docs = defaultdict(list)
    doc2cluster = {}
    for i, docs in enumerate(old_paths):
        cluster2docs[i] = docs
        for doc in docs:
            doc2cluster[doc] = i
    print("Start merging")
    start = time.time()
    new_cluster2docs, new_doc2cluster = merge(cluster2docs, doc2cluster, adj_lst=adj_lst,
                                              cluster_size=args.cluster_size, top_k=args.top_k)
    print("Time used:", datetime.timedelta(seconds=time.time() - start))
    new_paths = []
    for cluster, docs in new_cluster2docs.items():
        new_paths.append(docs)
    get_path_stats(new_paths)
    new_path = args.path_file.replace('.json', f'_merged_csize{args.cluster_size}_k{args.top_k}.json')
    print("Saving to", new_path)
    json.dump(new_paths, open(new_path, 'w'))


if __name__ == "__main__":
    main()
