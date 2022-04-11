import torch
import torch.nn.functional as F
import torch_sparse
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, cached=True, positional=False, num_nodes=None):
        super(GCN, self).__init__()

        self.positional = positional

        if positional and num_nodes is None: raise Exception("positional requires num_nodes")
        if positional:
            self.pos_embedding = torch.nn.Embedding(num_nodes, hidden_channels)
            in_channels += hidden_channels

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=cached))

        self.dropout = dropout

    def reset_parameters(self):
        if self.positional: torch.nn.init.xavier_uniform_(self.pos_embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        if self.positional: x = torch.cat([x, self.pos_embedding.weight], dim=1)
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SVD(torch.nn.Module):
    def __init__(self, adj_t, out_channels):
        super(SVD, self).__init__()
        self.adj_t = adj_t
        self.out_channels = out_channels
        self.embedding, _, _ = torch.svd_lowrank(self.adj_t.to_torch_sparse_coo_tensor(), q=self.out_channels)

    def reset_parameters(self):
        self.embedding, _, _ = torch.svd_lowrank(self.adj_t.to_torch_sparse_coo_tensor(), q=self.out_channels)

    def forward(self, x, adj_t):
        return self.embedding


class MCSVD(torch.nn.Module):
    def __init__(self, adj_t, out_channels, num_nodes, nsamples=1):
        super(MCSVD, self).__init__()
        self.adj_t = adj_t
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.nsamples = nsamples
        self.lin1 = torch.nn.Linear(out_channels, out_channels)
        self.lin2 = torch.nn.Linear(out_channels, out_channels)

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t):
        x = 0
        for _ in range(self.nsamples):
            perm = torch.randperm(self.num_nodes)
            adj_t = torch_sparse.permute(adj_t, perm)
            embedding, _, _ = torch.svd_lowrank(adj_t.to_torch_sparse_coo_tensor(), q=self.out_channels, niter=1)
            inv_perm = [None] * self.num_nodes
            for i, j in enumerate(perm):
                inv_perm[j.item()] = i
            embedding = embedding[inv_perm]
            embedding = F.relu(self.lin1(embedding))
            x += embedding
        x = x / self.nsamples
        x = F.relu(self.lin2(x))
        return x
