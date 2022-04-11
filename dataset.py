import torch
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling


class Dataset:
    def __init__(self, name, split, window, interventions=False):
        print("Initializing dataset")
        self.name = name
        self.split = split
        self.interventions = interventions
        if self.name[:4] == "ogbl":
            self.build_ogb()
        else:
            raise Exception("dataset not implemented")

        self.window = window

    def build_ogb(self):
        print("Building ogb")
        dataset = PygLinkPropPredDataset(name=self.name, root=self.name,
                                         transform=T.ToSparseTensor(remove_edge_index=False))
        print("Dataset:")
        print(dataset)
        data = dataset[0]
        print(data)

        self.split_edge = dataset.get_edge_split()
        self.adj_t = data.adj_t
        self.num_nodes = self.adj_t.sparse_sizes()[0]
        if data.x is None:
            self.x = torch.ones((self.num_nodes, 1))
        else:
            self.x = data.x.float()
        if self.interventions:
            raise Exception("interventions not implemented for this dataset")

        # Add negative edges in train data
        pos_edge_index = self.split_edge["train"]["edge"].t()
        neg_edge_index = negative_sampling(pos_edge_index, self.num_nodes, method='sparse')
        self.split_edge["train"]["edge_neg"] = neg_edge_index.t()

        # Concat positive and negative for later
        neg = self.split_edge[self.split]["edge_neg"]
        pos = self.split_edge[self.split]["edge"]
        self.edges = torch.cat([neg, pos])

    def __getitem__(self, idx):
        neg = self.split_edge[self.split]["edge_neg"]
        ids = torch.arange(idx * self.window, idx * self.window + self.window)
        ys = (ids >= len(neg)).view(-1, 1).float()

        return Data(x=self.x, adj_t=self.adj_t, edge=self.edges, idx=ids, y=ys)

    def __len__(self):
        return len(self.edges) // self.window

    def shuffle(self):
        neg = self.split_edge[self.split]["edge_neg"]
        pos = self.split_edge[self.split]["edge"]

        neg = neg[torch.randperm(neg.size(0))]
        pos = pos[torch.randperm(pos.size(0))]
        self.edges = torch.cat([neg, pos])


class Collater:
    def __call__(self, datas):
        tmp = datas[0]

        ys = torch.vstack([data.y for data in datas])
        ids = torch.cat([data.idx for data in datas])

        edges = tmp.edge[ids].t()

        return Data(x=tmp.x, adj_t=tmp.adj_t, edge=edges, y=ys)


if __name__ == "__main__":
    dataset = Dataset("ogbl-ddi", "Hits@20", "train")
    l = [dataset[0], dataset[1]]
    coll = Collater()
    coll(l)

    import pdb;

    pdb.set_trace()
