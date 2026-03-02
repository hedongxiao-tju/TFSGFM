from generate_few_shot_examples import create_few_data_folder
from utils.Classifier import Classifier
from utils.seed import set_seed
from utils.get_train_test_data import get_data
import numpy as np
import torch.optim as optim
import sys
import torch
import argparse
import warnings
from model import DualGCN, center_away_loss, neighbor_close_loss


def main():
    def train():
        model.train()
        optimizer.zero_grad()
        z1,z2 = model(train_data.x1, train_data.edge_index, train_data.x2, train_data.edge_index2, args, train = 1)

        loss11 = center_away_loss(z1)
        loss12 = neighbor_close_loss(z1,train_data.edge_index)
        loss21 = center_away_loss(z2)
        loss22 = neighbor_close_loss(z2,train_data.edge_index2)
        loss_train = loss11 + args.loss_ratio * loss12 + loss21 + args.loss_ratio * loss22

        loss_train.backward()
        optimizer.step()
        # print('Epoch: {:04d}'.format(epoch + 1),'loss_train: {:.4f}'.format(loss_train.item()))
    def test():
        model.eval()
        with torch.no_grad():
            z = model(test_data.x1, test_data.edge_index, test_data.x2, test_data.edge_index2, args, train = 0)

        log = Classifier(ft_in=z.shape[1], nb_classes=torch.max(test_data.y).item() + 1)
        num_trials =500
        # create_few_data_folder(num_trials, args.test_dataset, test_data, torch.max(test_data.y).item() + 1)

        shot_num = args.shot_num
        acc_list = []
        for i in range(num_trials):
            sample_data_foler_path = "./Experiment/sample_data/Node/{}/{}_shot/{}".format(args.test_dataset, shot_num, i + 1)
            idx_train = torch.load(f"{sample_data_foler_path}/train_idx.pt").type(torch.long).to('cuda')
            idx_test = torch.load(f"{sample_data_foler_path}/test_idx.pt").type(torch.long).to('cuda')
            log.forward(z[idx_train], test_data.y[idx_train], train=1).float().cuda()
            batch_size = 500
            total_correct = 0
            total_samples = 0
            for i in range(0, len(idx_test), batch_size):
                batch_indices = idx_test[i:i + batch_size]
                logits = log.forward(z[batch_indices], test_data.y[batch_indices]).float().cuda()
                preds = torch.argmax(logits, dim=1).cuda()
                total_correct += torch.sum(preds == test_data.y[batch_indices]).item()
                total_samples += len(batch_indices)
            acc = total_correct / total_samples
            acc_list.append(acc)
        average_acc = sum(acc_list) / num_trials
        print(f"Average ACC over {num_trials} trials: {average_acc:.4f}")
        std_acc = np.std(acc_list)
        print(f"Standard Deviation of ACC over {num_trials} trials: {std_acc:.4f}")
        print("-------------------------------------------------------------------------------------")

    train_dataset, test_data = get_data(args)
    model = DualGCN(test_data.x1.shape[1], test_data.x2.shape[1], args.hidden_dimension, args.out_dimension, args.GCN_num, args.GCN_num)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    for train_data in train_dataset: train_data.cuda()
    for epoch in range(args.epochs):
        for train_data in train_dataset:
            train()
    for train_data in train_dataset: train_data.detach().cpu()

    optimizer.zero_grad()
    test_data.cuda()
    test()
    test_data.detach().cpu()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='seed.')

    parser.add_argument('--shot_num', type=int, default=1, help='shot_num')

    parser.add_argument('--lr', type=float, default=0.00001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--loss_ratio', type=float, default=0.1, help='Proportion of loss composition')

    parser.add_argument('--top_k', type=int, default=10, help='top_k to build link')

    parser.add_argument('--svd_k', type=int, default=256, help='SVD_out_dimension')
    parser.add_argument('--hidden_dimension', type=int, default=256, help='GCN_hidden_dimension')
    parser.add_argument('--out_dimension', type=int, default=256, help='GCN_final_out_dimension')
    
    parser.add_argument('--GCN_num', type=int, default=2, help='GCN_layer_number')
    parser.add_argument('--DAD', type=int, default=5, help='DAD_layer_number')
    
    parser.add_argument('--split', type=int, default=4, help='split_ratio')


    parser.add_argument('--datasets_dir', type=str, default='datasets', help='datasets dir.')
    parser.add_argument('--train_datasets', type=str, nargs='+',default=[ 'Cora','CiteSeer', 'PubMed','Photo'], help='train datasets.')
    parser.add_argument('--test_dataset', type=str,default= 'Computers', help='test dataset.')


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    set_seed(args.seed)
    main();