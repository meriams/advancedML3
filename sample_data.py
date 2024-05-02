import networkx as nx 
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from collections import Counter
import random
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_training_values():
    mutag = TUDataset(root='./data/', name='MUTAG')

    train_loader = DataLoader(mutag, batch_size=1)
    node_list = []
    probability_list = []
    max_nodes = 0

    for data in train_loader:
        max_nodes = max(max_nodes, data.num_nodes)

        total_possible_edges = data.num_nodes * (data.num_nodes - 1) / 2
        probability = len(data.edge_index[0]) / total_possible_edges
        node_list.append(data.num_nodes)
        probability_list.append(probability)

    return node_list, probability_list, max_nodes


def get_training_graph():
    mutag = TUDataset(root='./data/', name='MUTAG')

    train_loader = DataLoader(mutag, batch_size=1)
    graph_list = []
    max_nodes = 0
    for data in train_loader:
        max_nodes = max(max_nodes, data.num_nodes)

    for data in train_loader:
        
        G = nx.Graph()

        for i in range(data.edge_index.shape[1]):
            source = data.edge_index[0, i].item()  # source node index
            target = data.edge_index[1, i].item()  # target node index
            G.add_edge(source, target)

        graph_list.append(G)
    return graph_list

def get_hashes_from_graphs(graphs):
    wl_list = []
    for g in graphs:
        wl_list.append(nx.weisfeiler_lehman_graph_hash(g))
    return wl_list

def sample_evaluation(graph, training_hashes):
    hashes = get_hashes_from_graphs(graph)
    total_length = len(hashes)

    training_hash_set = set(training_hashes)

    hash_counter = Counter(hashes)

    novel = sum(1 for h in hashes if h not in training_hash_set)
    unique = sum(1 for h, count in hash_counter.items() if count == 1)
    novel_unique = sum(1 for h, count in hash_counter.items() if h not in training_hash_set and count == 1)

    return float(novel / total_length), float(unique / total_length), float(novel_unique / total_length)

def full_evaluation(baseline_graph, vae_graph, training_hashes):
    baseline_data = sample_evaluation(baseline_graph, training_hashes)
    vae_data = sample_evaluation(vae_graph, training_hashes)
    return baseline_data, vae_data

def graph_statistics(Graphs):
    node_degree = []
    clustering_coefficient = []
    eigenvector_centrality = []
    for i in range(len(Graphs)):
        G = Graphs[i]
        node_degree.append(G.degree)
        clustering_coefficient.append(nx.clustering(G))
        eigenvector_centrality.append(nx.eigenvector_centrality(G, max_iter=10000))
    return node_degree, clustering_coefficient, eigenvector_centrality

def get_buckets(node_degree, clustering_coefficient, eigenvector_centrality):
    degree_bucket = []
    clustering_bucket = []
    eigenvector_bucket = []
    for index in range(len(node_degree)):
        for val in node_degree[index]:
            degree_bucket.append(val[1])
        for val in clustering_coefficient[index].items():
            clustering_bucket.append(val[1])
        for val in eigenvector_centrality[index].items():
            eigenvector_bucket.append(val[1])
    return degree_bucket, clustering_bucket, eigenvector_bucket

def get_min_max(val_list):
    max_val = 0
    min_val = int(1e9)
    for val in val_list:
        max_val = max(max_val, val)
        min_val = min(min_val, val)
    return max_val, min_val


def show_graphs(degree_list, clustering_list, eigenvector_list, model_text):
    degree_bucket, clustering_bucket, eigenvector_bucket = get_buckets(degree_list, clustering_list, eigenvector_list)
    data = [degree_bucket, clustering_bucket, eigenvector_bucket]
    titles = ['Node Degree', 'Clustering Coefficient', 'Eigenvector Centrality']
    for i, d in enumerate(data):
        plt.figure()
        plt.title(f"{model_text} - {titles[i]}")
        plt.hist(d, bins=20)
        plt.xlabel('Interval')
        plt.xlabel('Amount')
        plt.savefig(f"plots/{model_text}-{titles[i]}.png")
        plt.show()

def main():
    training_node_list, training_probability_list, max_nodes = get_training_values()
    max_index = len(training_node_list) - 1
    baseline_graphs = []

    for i in range(1000):
        ran_index = random.randint(0,max_index)

        nodes = training_node_list[ran_index]
        indexes = [index for index, value in enumerate(training_node_list) if value == nodes]

        selected_probabilities = [training_probability_list[index] for index in indexes]
        probability = np.mean(selected_probabilities)

        temp_G = nx.erdos_renyi_graph(nodes, p=probability)
        temp_G.remove_nodes_from(list(nx.isolates(temp_G)))
        baseline_graphs.append(temp_G)

    vae_graphs = []
    for i in range(1000):
        graph = nx.read_adjlist(f"graphs/graph_{i}.adjlist")
        vae_graphs.append(graph)

    training_graph = get_training_graph()
    training_hashes = get_hashes_from_graphs(training_graph)
    # evaluation = sample_evaluation(baseline_graphs, training_hashes)
    novelty_results = full_evaluation(baseline_graphs, vae_graphs, training_hashes)
    print(novelty_results)
    training_node_degree, training_clustering_coefficient, training_eigenvector_centrality = graph_statistics(training_graph)
    base_node_degree, base_clustering_coefficient, base_eigenvector_centrality = graph_statistics(baseline_graphs)
    vae_node_degree, vae_clustering_coefficient, vae_eigenvector_centrality = graph_statistics(vae_graphs)
    show_graphs(training_node_degree, training_clustering_coefficient, training_eigenvector_centrality, "Training")
    show_graphs(base_node_degree, base_clustering_coefficient, base_eigenvector_centrality, "Baseline")
    show_graphs(vae_node_degree, vae_clustering_coefficient, vae_eigenvector_centrality, "VAE")


main()