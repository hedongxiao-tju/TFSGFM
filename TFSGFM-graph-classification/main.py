
import numpy as np
from utils.seed import set_seed
from utils.get_train_test_data import get_data
import torch.optim as optim
import sys
import torch
import argparse
import warnings
from model import DualGCN, center_away_loss, neighbor_close_loss
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch_geometric.nn as pyg_nn
from sklearn.preprocessing import normalize
from torch_geometric.loader import DataLoader
def main():

    def train():
        model.train()
        for data in train_loader:
            data = data.cuda()
            optimizer.zero_grad()
            z1,z2 = model(data.x1, data.edge_index, data.x2, data.edge_index2)

            loss11 = center_away_loss(z1)
            loss12 = neighbor_close_loss(z1,data.edge_index)
            loss21 = center_away_loss(z2)
            loss22 = neighbor_close_loss(z2,data.edge_index2)
            loss_train = loss11 + args.loss_ratio * loss12 + loss21 + args.loss_ratio * loss22

            loss_train.backward()
            optimizer.step()

    def test():
        model.eval()
        x_list = []
        y_list = []
        with torch.no_grad():
            for i, batch_g in enumerate(test_loader):
                batch_g = batch_g.cuda()
                z1,z2 = model(batch_g.x1, batch_g.edge_index, batch_g.x2, batch_g.edge_index2)
                z = torch.cat([z1, z2], dim=1)
                out = pyg_nn.global_mean_pool(z, batch_g.batch)
                y_list.append(batch_g.y.cpu().numpy())
                x_list.append(out.cpu().numpy())
        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        num_repeats = 50
        acc_list = []
        f1_list = []
        for _ in range(num_repeats):
            x_proto, x_test, y_proto, y_test = train_test_split(x, y, test_size=0.8, stratify=y)
            unique_classes = np.unique(y)
            prototypes = {}
            for cls in unique_classes:
                prototypes[cls] = np.mean(x_proto[y_proto == cls], axis=0)
            for cls in prototypes:
                prototypes[cls] = normalize(prototypes[cls].reshape(1, -1))[0]
            x_test_norm = normalize(x_test)
            predictions = []
            for x_vec in x_test_norm:
                similarities = {cls: np.dot(x_vec, prototypes[cls]) for cls in prototypes}
                pred_cls = max(similarities, key=similarities.get)
                predictions.append(pred_cls)
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='macro')
            acc_list.append(acc)
            f1_list.append(f1)
        acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
        f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
        print(f"Average Accuracy: {acc_mean:.4f}, Standard Deviation: {acc_std:.6f}")
        print(f"Average F1-score: {f1_mean:.4f}, Standard Deviation: {f1_std:.6f}")
        print("-------------------------------------------------------------------------------------")

    train_dataset, test_data = get_data(args)
    model = DualGCN(test_data[0].x1.shape[1], test_data[0].x2.shape[1], args.hidden_dimension, args.out_dimension, args.GCN_num, args.GCN_num)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        for train_data in train_dataset:
            train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
            train()

    optimizer.zero_grad()
    test_loader = DataLoader(test_data, batch_size=128)
    test()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='seed.')

    parser.add_argument('--lr', type=float, default=0.000001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--loss_ratio', type=float, default=0.1, help='Proportion of loss composition')

    parser.add_argument('--top_k', type=int, default=1, help='top_k to build link')
    parser.add_argument('--svd_k', type=int, default=512, help='SVD_out_dimension')
    parser.add_argument('--hidden_dimension', type=int, default=256, help='GCN_hidden_dimension')
    parser.add_argument('--out_dimension', type=int, default=128, help='GCN_final_out_dimension')
    
    parser.add_argument('--GCN_num', type=int, default=2, help='GCN_layer_number')

    parser.add_argument('--split', type=int, default=4, help='split_ratio')


    parser.add_argument('--datasets_dir', type=str, default='datasets', help='datasets dir.')
    parser.add_argument('--train_datasets', type=str, nargs='+',default=['IMDB-BINARY' ,'ENZYMES'], help='train datasets.')
    parser.add_argument('--test_dataset', type=str,default=  'DD' ,  help='test dataset.')



    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    set_seed(args.seed)
    main();