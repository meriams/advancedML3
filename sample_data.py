import networkx as nx 
import matplotlib.pyplot as plt 
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def get_training_values():
    mutag = TUDataset(root='./data/', name='MUTAG')

    train_loader = DataLoader(mutag, batch_size=1)
    total_possible_edges = 0
    total_edges = 0
    max_nodes = 0
    for data in train_loader:
        max_nodes = max(max_nodes, data.num_nodes)

    for data in train_loader:
        for e in range(max_nodes):
            total_possible_edges += e
        total_edges += len(data.edge_index[0])

    return total_possible_edges, total_edges


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
    
    novel = 0
    unique = 0
    novel_unique = 0
    for i, b_hash in enumerate(hashes):
        temp_hashes = hashes.copy()
        temp_hashes.pop(i)
        crit1 = False
        crit2 = False
        if b_hash not in training_hashes:
            novel += 1
            crit1 = True
        if b_hash not in temp_hashes:
            unique += 1
            crit2 = True
        if crit1 and crit2:
            novel_unique += 1
    return float(novel / total_length), float(unique / total_length), float(novel_unique / total_length)

def full_evaluation(baseline_graph, vae_graph, training_hashes):
    baseline_data = sample_evaluation(baseline_graph, training_hashes)
    vae_data = sample_evaluation(vae_graph, training_hashes)
    return baseline_data, vae_data

def graph_statistics(Graphs):
    node_degree = []
    clustering_coefficient = []
    eigenvector_centrality = []
    for i in range(1000):
        G = Graphs[i]
        node_degree.append(G.degree)
        clustering_coefficient.append(nx.clustering(G))
        eigenvector_centrality.append(nx.eigenvector_centrality(G, max_iter=10000))
    return node_degree, clustering_coefficient, eigenvector_centrality

def show_graphs(degree_graph, clustering_graph, eigenvector_graph):
    data = [degree_graph, clustering_graph, eigenvector_graph]
    titles = ['Node Degree', 'Clustering Coefficient', 'Eigenvector Centrality']
    for i, d in enumerate(data):
        plt.figure()
        plt.title(titles[i])
        plt.hist(d, bins=20)
        plt.xlabel('Interval')
        plt.xlabel('Amount')
        plt.show()
def main():
    baseline_graph = [nx.erdos_renyi_graph(28, p=(7442 / 71064)) for _ in range(1000)]
    vae_graph = [nx.erdos_renyi_graph(28, p=(7442 / 71064)) for _ in range(1000)]
    training_graph = get_training_graph()
    training_hashes = get_hashes_from_graphs(training_graph)
    evaluation = sample_evaluation(baseline_graph, training_hashes)
    novelty_results = full_evaluation(baseline_graph, vae_graph, training_hashes)
    node_degree, clustering_coefficient, eigenvector_centrality = graph_statistics(baseline_graph)
    # show_graphs(node_degree, clustering_coefficient, eigenvector_centrality)
