import argparse
import torch

import numpy as np
from ogb.linkproppred import Evaluator as Evaluator_
from ogb.linkproppred import PygLinkPropPredDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PygDataloader

from SEALdataset import SEALDataset, SEALDynamicDataset
from dataset import Dataset, Collater
from embedding import *
from link import *
from models import DGCNN, SimpleModel

import warnings
warnings.simplefilter("ignore", UserWarning)


class Evaluator:
    def __init__(self, name, eval_method):
        self.name = name
        self.eval_method = eval_method

    def __call__(self, pos, neg):
        if self.eval_method[:4] == "Hits":
            evaluator = Evaluator_(name=self.name)
            K = int(self.eval_method.split("@")[1])
            evaluator.K = K
            result = evaluator.eval({
                'y_pred_pos': pos,
                'y_pred_neg': neg,
            })[f'hits@{K}']
            return result
        else:
            raise Exception("eval_method not implemented")


def train(loader, model, optimizer, loss_fn, device, clip_grad=True):
    model.train()

    total_loss = total_examples = 0

    data: Data
    for data in loader:

        data = data.to(device)

        optimizer.zero_grad()

        out = model(data)

        y = data.y.view(out.shape).to(torch.float)
        loss = loss_fn(out, y)
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * data.y.shape[0]
        total_examples += data.y.shape[0]

    return total_loss / total_examples


@torch.no_grad()
def test(loader, model, evaluator, device):
    model.eval()

    pos_preds = []
    neg_preds = []

    data: Data
    for data in loader:
        data = data.to(device)

        out = model(data)

        y = data.y.view(out.shape).to(torch.float)
        pos_preds.append(out[y == 1.].cpu())
        neg_preds.append(out[y == 0.].cpu())

    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0)

    return evaluator(pos_pred, neg_pred)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    parser = argparse.ArgumentParser(description='Link prediction tasks')
    parser.add_argument('--dataset', type=str, default="ogbl-ppa")
    parser.add_argument('--node_embedding', type=str, default="GCN")  # SVD,GCN
    parser.add_argument('--eval_method', type=str, default="Hits@100")  # Hits@K
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=70000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--factor', type=int, default=0.5)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--positional', action='store_true')  # Force symmetric model (GNN) to be positional

    parser.add_argument('--seal', action='store_true')
    parser.add_argument('--dynamic_train', action='store_true',
                        help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--use_feature', action='store_true',
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--train_percent', type=float, default=100)
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.seal:
        dataset = PygLinkPropPredDataset(name=args.dataset)
        data = dataset[0]
        path = dataset.root + '_seal'
        split_edge = dataset.get_edge_split()

        dataset_class = SEALDynamicDataset if args.dynamic_train else SEALDataset
        dataset = dataset_class(
            path,
            data,
            split_edge,
            num_hops=1,
            percent=args.train_percent,
            split='train',
            ratio_per_hop=args.ratio_per_hop,
            max_nodes_per_hop=args.max_nodes_per_hop,
        )

        valid_dataset = SEALDynamicDataset(
            path,
            data,
            split_edge,
            num_hops=1,
            split='valid',
            ratio_per_hop=args.ratio_per_hop,
            max_nodes_per_hop=args.max_nodes_per_hop,
        )

        test_dataset = SEALDynamicDataset(
            path,
            data,
            split_edge,
            num_hops=1,
            split='test',
            ratio_per_hop=args.ratio_per_hop,
            max_nodes_per_hop=args.max_nodes_per_hop,
        )

        train_loader = PygDataloader(dataset, args.batch_size, shuffle=True, num_workers=16)
        valid_loader = PygDataloader(valid_dataset, args.batch_size, num_workers=16)
        test_loader = PygDataloader(test_dataset, args.batch_size, num_workers=16)

        model = DGCNN(args.hidden_channels, args.num_layers, train_dataset=dataset,
                      dynamic_train=args.dynamic_train, use_feature=args.use_feature).to(device)

    else:
        window = 100
        args.batch_size = args.batch_size // window
        dataset = Dataset(args.dataset, split="train", window=window)
        valid_dataset = Dataset(args.dataset, split="valid", window=window)
        test_dataset = Dataset(args.dataset, split="test", window=window)

        train_loader = DataLoader(dataset, args.batch_size, collate_fn=Collater(), shuffle=True)
        valid_loader = DataLoader(valid_dataset, args.batch_size, collate_fn=Collater())
        test_loader = DataLoader(test_dataset, args.batch_size, collate_fn=Collater())

        if args.node_embedding == "GCN":
            node_embedding = GCN(dataset.x.size(1), args.hidden_channels, args.hidden_channels, args.num_layers,
                                 args.dropout, positional=args.positional, num_nodes=dataset.num_nodes)
        elif args.node_embedding == "SVD":
            node_embedding = SVD(dataset.adj_t, args.hidden_channels)
        elif args.node_embedding == "MCSVD":
            node_embedding = MCSVD(dataset.adj_t, args.hidden_channels, dataset.num_nodes, nsamples=1)
        else:
            raise Exception("node_embedding not implemented")

        predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                                  args.num_layers, args.dropout)

        model = SimpleModel(node_embedding, predictor).to(device)

    results = []
    for run in range(args.runs):
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None if args.seal else ReduceLROnPlateau(optimizer, mode='max', factor=args.factor,
                                                             patience=args.patience, min_lr=0.00001)
        loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
        evaluator = Evaluator(args.dataset, args.eval_method)

        best_valid = final_result = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(train_loader, model, optimizer, loss_fn, device, clip_grad=not args.seal)
            if not args.seal: dataset.shuffle()

            if scheduler is not None or epoch % args.eval_steps == 0:
                valid_result = test(valid_loader, model, evaluator, device)
                test_result = test(test_loader, model, evaluator, device)

                if scheduler is not None: scheduler.step(valid_result)

                if get_lr(optimizer) == 0.00001: break

                print("Epoch:\t", epoch, "Loss:\t", loss, "Valid:\t", valid_result, "Test:\t", test_result, "LR:\t",
                      get_lr(optimizer))
                if valid_result > best_valid:
                    best_valid = valid_result
                    final_result = test_result

        results.append(final_result)
        print("RUN\t", run, results)

    print("Final result:\t", np.array(results).mean(), np.array(results).std())


if __name__ == "__main__":
    main()
