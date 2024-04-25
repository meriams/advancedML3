import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# Load the MUTAG dataset
mutag = TUDataset(root='./data/', name='MUTAG')

train_loader = DataLoader(mutag, batch_size=1)

wl_list = []

for data in train_loader:
    G = nx.Graph()

    for i in range(data.edge_index.shape[1]):
        source = data.edge_index[0, i].item()  # source node index
        target = data.edge_index[1, i].item()  # target node index
        G.add_edge(source, target)

    wl_list.append(nx.weisfeiler_lehman_graph_hash(G))



