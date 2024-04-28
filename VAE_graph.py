import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch.distributions as td
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import numpy as np
from tqdm import tqdm
import networkx as nx


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim):
        """
        Initialize the Gaussian encoder for graph data to encode the entire graph
        into a single latent representation.
        """
        super(GaussianEncoder, self).__init__()
        self.relu = nn.ReLU()

        self.gcn1 = GCNConv(num_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.pool = global_mean_pool

        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_std = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch_index):
        """
        Forward pass of the encoder, producing a Gaussian distribution for the entire graph.
        """
        x = self.relu(self.gcn1(x, edge_index))
        x = self.relu(self.gcn2(x, edge_index))
        x = self.pool(x, batch_index)

        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std)

        return td.Independent(td.Normal(loc=mean, scale=std), 1)

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, max_nodes):
        super(GraphDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.fc1 = nn.Linear(latent_dim, max_nodes * max_nodes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        logits = self.fc1(z)
        logits = logits.view(-1, self.max_nodes, self.max_nodes)

        # Symmetrize the logits
        logits = (logits + logits.transpose(-2, -1)) / 2

        # Ensure no edge from node to itself
        indices = torch.arange(0, self.max_nodes)
        logits[:, indices, indices] = -float('inf')

        probabilities = self.sigmoid(logits)

        return probabilities


class VAE(nn.Module):
    def __init__(self, prior, encoder, decoder, max_nodes):
        super(VAE, self).__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.max_nodes = max_nodes

    def forward(self, features, edge_index, batch):
        q_z = self.encoder(features, edge_index, batch)
        z = q_z.rsample()
        probabilities = self.decoder(z)
        p_x = td.Independent(td.Bernoulli(probabilities), 3)

        adj = to_dense_adj(edge_index, batch=batch, max_num_nodes=self.max_nodes)
        log_prob = p_x.log_prob(adj).sum()

        # KL Divergence
        p_z = self.prior()
        kl_div = td.kl_divergence(q_z, p_z).mean()

        elbo = log_prob - kl_div
        return -elbo

    def sample(self, n_samples=1):
        z = self.prior().sample((n_samples,))
        probabilities = self.decoder(z)
        samples = (probabilities > 0.5).int()
        return samples

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()
    num_steps = len(data_loader) * epochs
    epoch = 0

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            data = next(iter(data_loader))[0]
            data = data.to(device)
            optimizer.zero_grad()
            loss = model(data.x, data.edge_index, data.batch)
            loss.backward()
            optimizer.step()

            # Report
            if step % 1 == 0:
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

            if (step + 1) % len(data_loader) == 0:
                epoch += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'],
                        help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='VAE_graph.pt',
                        help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png',
                        help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='torch device (default: %(default)s)')
    parser.add_argument('--batch', type=int, default=16, metavar='N',
                        help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=25000, metavar='N',
                        help='number of epochs to train (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load the dataset
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG', use_edge_attr=False)
    data_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    max_nodes = np.max([data.num_nodes for data in dataset])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize
    input_dim = 7
    hidden_dim = 64
    latent_dim = 32
    device = args.device
    epochs = args.epochs

    # Initialize the VAE
    encoder = GaussianEncoder(input_dim, hidden_dim, latent_dim).to(device)
    decoder = GraphDecoder(latent_dim, max_nodes)
    prior = GaussianPrior(latent_dim).to(device)
    model = VAE(prior, encoder, decoder, max_nodes).to(device)

    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train model
        train(model, optimizer, data_loader, epochs, device)

        # Save model
        torch.save(model.state_dict(), args.model)

    if args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        model.eval()
        with torch.no_grad():
            samples = (model.sample((1000))).cpu()
            for idx in range(samples.shape[0]):
                sample = samples[idx]
                sample = torch.triu(sample, diagonal=0)
                edge_index = dense_to_sparse(sample)[0]

                G = nx.Graph()

                for i in range(edge_index.shape[1]):
                    source = edge_index[0, i].item()  # source node index
                    target = edge_index[1, i].item()  # target node index
                    G.add_edge(source, target)

                nx.write_adjlist(G, f"graphs/graph_{idx}.adjlist")