import heapq
import random
from collections import defaultdict

import numpy as np
import tqdm


class Node:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.neighbors = []

    def add_neighbor(self, neighbor, weight: float = 1.0):
        self.neighbors.append((neighbor, weight))

    def get_neighbors(self):
        return self.neighbors

    def get_doc_id(self):
        return self.doc_id

    def get_out_degree(self):
        return len(self.neighbors)

    def __str__(self):
        return self.doc_id

    def __lt__(self, other):
        return self.doc_id < other.doc_id


class Graph:
    def __init__(self, use_heap=True, degree_measure="all"):
        self.use_heap = use_heap
        self.nodes = {}
        self.min_heap = []
        if self.use_heap:
            self.get_min_degree_node = self.get_min_degree_node_heap
            self.get_max_degree_node = self.get_max_degree_node_heap
        else:
            self.get_min_degree_node = self.get_min_degree_node_no_heap

        self.degree_measure = degree_measure
        self.in_degrees = {}
        self.init_degree_measure()

    def init_degree_measure(self):
        if self.degree_measure == "all":
            self.get_node_degree = self.get_node_in_and_out_degree
        elif self.degree_measure == "in":
            self.get_node_degree = self.get_node_in_degree
        elif self.degree_measure == "out":
            self.get_node_degree = self.get_node_out_degree
        else:
            raise ValueError("Invalid degree measure")

    def build_shuffled_nodes_lst(self):
        self.random_nodes_id_lst = list(self.nodes.keys())
        random.shuffle(self.random_nodes_id_lst)
        print("Initialized shuffled nodes list with", len(self.random_nodes_id_lst), "nodes")

    def get_random_node(self) -> None:
        # get the next random node that has not been deleted
        while self.random_nodes_id_lst:
            doc_id = self.random_nodes_id_lst.pop()
            if doc_id in self.nodes:
                return self.nodes[doc_id]
        return None

    def build_min_heap(self):
        if self.degree_measure in ['in', 'all']:
            # If the degree measure is in or all, we can use the in degrees to build the heap
            self.get_all_nodes_in_degree()

        # Build the heap once after all nodes and edges have been added
        self.min_heap = [(self.get_node_degree(node.get_doc_id()), node) for node in self.nodes.values()]
        heapq.heapify(self.min_heap)
        print("Initialized min heap with", len(self.min_heap), "nodes")
        # print the statistics of the nodes
        degrees = [self.get_node_degree(node) for node in self.nodes]
        print("Min degree:", min(degrees))
        print("Max degree:", max(degrees))
        print("Average degree:", sum(degrees) / len(degrees))
        print("Standard deviation of degrees:", np.std(degrees))

    def build_max_heap(self):
        if self.degree_measure in ['in', 'all']:
            # If the degree measure is in or all, we can use the in degrees to build the heap
            self.get_all_nodes_in_degree()

        # Build the heap once after all nodes and edges have been added
        self.max_heap = [(-self.get_node_degree(node.get_doc_id()), node) for node in self.nodes.values()]
        heapq.heapify(self.max_heap)
        print("Initialized max heap with", len(self.max_heap), "nodes")
        # print the statistics of the nodes
        degrees = [self.get_node_degree(node) for node in self.nodes]
        print("Min degree:", min(degrees))
        print("Max degree:", max(degrees))
        print("Average degree:", sum(degrees) / len(degrees))
        print("Standard deviation of degrees:", np.std(degrees))

    def get_min_degree_node_no_heap(self):
        min_degree_node = min(self.nodes.values(), key=lambda node: node.get_degree(), default=None)
        return min_degree_node.get_doc_id() if min_degree_node is not None else None

    def get_min_degree_node_heap(self) -> Node:
        # Efficiently get the node with the minimum degree
        while self.min_heap:
            degree, node = heapq.heappop(self.min_heap)
            if node.get_doc_id() in self.nodes:
                return node
        return None

    def get_max_degree_node_heap(self) -> Node:
        # Efficiently get the node with the maximum degree
        while self.max_heap:
            degree, node = heapq.heappop(self.max_heap)
            if node.get_doc_id() in self.nodes:
                return node
        return None

    def add_node(self, node: Node):
        self.nodes[node.get_doc_id()] = node

    def add_edge(self, node1: Node, node2: Node, weight: float, directed: bool):
        if directed:
            # if directed, only add the edge from node1 to node2
            node1.add_neighbor(node2, weight)
        else:
            # if undirected, add the edge from node1 to node2 and from node2 to node1
            node1.add_neighbor(node2, weight)
            node2.add_neighbor(node1, weight)

    def check_edge_weight(self, node1: Node, node2: Node):
        for neighbor in node1.get_neighbors():
            if neighbor[0] == node2:
                return neighbor[1]
        return -1

    def get_nodes(self):
        return self.nodes

    def get_node(self, doc_id: str):
        return self.nodes[doc_id]

    def get_node_neighbors(self, doc_id: str) -> set:
        neighbors = set()
        for neighbor in self.nodes[doc_id].get_neighbors():
            neighbors.add(neighbor[0].get_doc_id())
        return neighbors

    def get_best_available_neighbor(self, doc_id: str):
        best_neighbor = None
        best_weight = -1
        for neighbor in self.nodes[doc_id].get_neighbors():
            if neighbor[0].get_doc_id() not in self.nodes:  # neighbor is already deleted
                continue
            if neighbor[1] > best_weight:
                best_neighbor = neighbor[0]
                best_weight = neighbor[1]
        return best_neighbor

    def get_node_out_degree(self, doc_id: str):
        return self.nodes[doc_id].get_out_degree()

    def get_node_in_degree(self, doc_id: str):
        # in degree should be precomputed
        return self.in_degrees[doc_id]

    def get_all_nodes_in_degree(self):
        # default value is 0
        self.in_degrees = defaultdict(int)
        # traverse the neighbors of all nodes to get the in degree of each node
        for node in tqdm.tqdm(self.nodes.values(), desc="Getting in degrees"):
            for neighbor in node.get_neighbors():
                self.in_degrees[neighbor[0].get_doc_id()] += 1
        return self.in_degrees

    def get_node_in_and_out_degree(self, doc_id: str):
        return self.get_node_in_degree(doc_id) + self.get_node_out_degree(doc_id)

    def delete_node(self, doc_id: str):
        del self.nodes[doc_id]

    def get_node_ids(self):
        return self.nodes.keys()

    def get_node_count(self):
        return len(self.nodes)

    def __str__(self):
        return str(self.nodes)

    def copy(self):
        new_graph = Graph()
        for node in self.nodes:
            new_graph.add_node(node)
        for node in self.nodes:
            for neighbor in node.get_neighbors():
                new_graph.add_edge(node, neighbor[0], neighbor[1])
        return new_graph

    def get_edges(self):
        edges = []
        for node in self.nodes.values():
            for neighbor in node.get_neighbors():
                edges.append((node, neighbor[0]))
        return edges
