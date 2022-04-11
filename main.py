import argparse
import torch
import numpy as np
from ogb.linkproppred import Evaluator as Evaluator_
from ogb.linkproppred import PygLinkPropPredDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PygDataloader
# import torch.profiler
from SEALdataset import SEALDataset, SEALDynamicDataset
from dataset import Dataset, Collater
from embedding import *
from link import *
from models import DGCNN, SimpleModel
# from torchsummary import summary
import warnings
import cProfile
warnings.simplefilter("ignore", UserWarning)
import time

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


def train(loader, model, optimizer, loss_fn, device, clip_grad=True, l1_strength=0, l2_strength=0):
    model.train()

    total_loss = total_examples = 0

    data: Data
    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()

        out = model(data)

        y = data.y.view(out.shape).to(torch.float)
        loss = loss_fn(out, y)
        
        # add l1 regularization to combat overfitting
        # l1_regularization = torch.tensor(0).float().to(device)
        l1_weight, l2_weight = l1_strength, l2_strength
      
        l1 = sum(p.abs().sum() for p in model.parameters()) if l1_weight > 0 else 0
        l2 = sum(p.pow(2.0).sum() for p in model.parameters()) if l2_weight > 0 else 0

        loss += l1_weight * l1 + l2_weight*l2
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
        pos_preds.append(out[y >= .5].cpu())
        neg_preds.append(out[y < .5].cpu())

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
    parser.add_argument('--l1_strength', type=float, default=0)
    parser.add_argument('--l2_strength', type=float, default=0)
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--stop_early', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name=args.dataset)
    data = dataset[0]
    path = dataset.root + '_seal'
    split_edge = dataset.get_edge_split()

    dataset_class = SEALDynamicDataset if args.dynamic_train else SEALDataset
    dataset = dataset_class(
        path,
        data,
        split_edge,
        num_hops=args.num_hops,
        percent=args.train_percent,
        split='train',
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        reverse=args.reverse,
    )
    
    valid_dataset = SEALDynamicDataset(
        path,
        data,
        split_edge,
        num_hops=args.num_hops,
        split='valid',
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        reverse=args.reverse,
    )
    
    test_dataset = SEALDynamicDataset(
        path,
        data,
        split_edge,
        num_hops=args.num_hops,
        split='test',
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        reverse=args.reverse,
    )

    train_loader = PygDataloader(dataset, args.batch_size, shuffle=True, num_workers=16)
    valid_loader = PygDataloader(valid_dataset, args.batch_size, num_workers=16)
    test_loader = PygDataloader(test_dataset, args.batch_size, num_workers=16)
    print("Loading Model")
    model = DGCNN(args.hidden_channels, args.num_layers, train_dataset=dataset,
                    dynamic_train=args.dynamic_train, use_feature=args.use_feature).to(device)


    train_results = []
    valid_results = []
    test_results = []
    best_epochs = []
    print("Running...")
    for run in range(args.runs):
        print("Run: ", run)
        start_time = time.time()
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None if args.seal else ReduceLROnPlateau(optimizer, mode='max', factor=args.factor,
                                                            patience=args.patience, min_lr=0.00001)
        loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
        evaluator = Evaluator(args.dataset, args.eval_method)

        best_validation = 0
        print(f'\tBegin Training:')
        for epoch in range(1, 1 + args.epochs):
            loss = train(train_loader, model, optimizer, loss_fn, device, clip_grad=not args.seal, l1_strength=args.l1_strength, l2_strength=args.l2_strength)
            if not args.seal: dataset.shuffle()
            print("\tEpoch: ", epoch, ", Loss: ", loss)
            if scheduler is not None or epoch % args.eval_steps == 0:
                
                # Early Stopping
                if args.stop_early:
                    validation = test(valid_loader, model, evaluator, device)
                    if validation >= best_validation:
                        best_validation = validation
                    else:
                        valid_results.append(best_validation)
                        break
                
                if args.save_best:
                    validation = test(valid_loader, model, evaluator, device)
                    if validation >= best_validation:
                        best_validation = validation
                        best_epoch = epoch
                        test_result = test(test_loader, model, evaluator, device)
                if get_lr(optimizer) == 0.00001: break


        training_result = test(train_loader, model, evaluator, device)
        train_results.append(training_result)
        if not args.save_best:
            test_result = test(test_loader, model, evaluator, device)
        else:
            best_epochs.append(best_epoch)
        test_results.append(test_result)
        print(f'Run finished in {(time.time()-start_time):.2f} seconds with training score of: {training_result} \t and test score of:\t {test_result}')

    #### End training Loop ####
    test_results = np.array(test_results)
    train_results = np.array(train_results)
    test_mean = test_results.mean()
    test_std = test_results.std()
    train_mean = train_results.mean()
    train_std = train_results.std()
    print("Final test Score:\tmean:", test_mean, ", std: ",test_std)
    print(f"Final train Score:\t mean: {train_mean}, std: {train_std}")
    if args.stop_early: 
        valid_results = np.array(valid_results)
        valid_mean = valid_results.mean()
        valid_std =  valid_results.std()
        print(f"Final validation Score:\t mean: {valid_mean}, std: {valid_std}")
    print(f"Amount of overfitting: {train_mean-test_mean}")
    if args.save_best:
        best_epochs = np.array(best_epochs)
        print(f"Best Epochs for validation: {best_epochs}, avg: {best_epochs.mean()}, std: {best_epochs.std()}")

if __name__ == "__main__":
    # cProfile.run('main()',"./profiling/dynamic_gen",1)
    main()
