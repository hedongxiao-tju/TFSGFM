from Dataset_Load import load_dataset
from torch_geometric.utils import to_undirected
from sklearn.decomposition import PCA
import numpy as np
import scipy as sp
import scipy.sparse as sp
import torch

def get_data(args):
    with torch.no_grad():
        train_dataset = []
        for dataset_name in args.train_datasets:
            train_data = load_dataset(dataset_name, args.datasets_dir,args)[0]
            print(train_data)
            print(torch.max(train_data.y).item() + 1)

            train_data.x = pca(train_data.x, args)
            train_data.x = x_sort(train_data.x, train_data.edge_index)
            train_data.x1, train_data.x2 = x_split(train_data.x, args)
            train_data.edge_index2 = edge_create(train_data.x2, args)
            train_dataset.append(train_data.detach().cpu())

        test_data = load_dataset(args.test_dataset, args.datasets_dir,args)[0]
        test_data.x = pca(test_data.x, args)
        test_data.x = x_sort_test(test_data.x, test_data.edge_index, args)
        test_data.x1, test_data.x2 = x_split(test_data.x, args)
        test_data.edge_index2 = edge_create(test_data.x2, args)
        test_data.detach().cpu()
    return train_dataset,test_data


def edge_create(x, args):

    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    sim_matrix = torch.matmul(x_norm, x_norm.t())

    num_nodes = x.size(0)
    topk = args.top_k

    _, topk_indices = torch.topk(sim_matrix, k=topk + 1, dim=-1)
    topk_indices = topk_indices[:, 1:]

    row_indices = torch.arange(num_nodes).unsqueeze(1).expand(-1, topk).reshape(-1)
    col_indices = topk_indices.reshape(-1)
    edge_index = torch.stack([row_indices, col_indices], dim=0)

    return edge_index

def x_sort(x, edge_index):
    edge_index = to_undirected(edge_index)

    src, dst = edge_index
    feat_diff = x[src] - x[dst]

    feature_similarity = (feat_diff ** 2).mean(dim=0)

    sorted_indices = torch.argsort(feature_similarity)

    x_reordered = x[:, sorted_indices]

    return x_reordered
def x_sort_test(x, edge_index, args):
    edge_index = to_undirected(edge_index)
    
    src, dst = edge_index
    feat_diff = x[src] - x[dst] 
    
    feature_similarity = (feat_diff ** 2).mean(dim=0)
    max_val = feature_similarity.max()
    min_val = feature_similarity.min()
    range_val = max_val - min_val
    if args.DAD == -1:
        args.DAD = 0 if range_val > 0.01 else 5

    sorted_indices = torch.argsort(feature_similarity)

    x_reordered = x[:, sorted_indices]

    return x_reordered

def x_split(x, args):
    num_features = x.size(1)
    mid = num_features // args.split
    x_front = x[:, :mid]
    x_back = x[:, mid:]
    return x_front, x_back

def pca(x, args):
    x = x.numpy()

    n_samples, n_features = x.shape
    if n_features < args.svd_k:
        pad_features = args.svd_k - n_features
        x = np.pad(x, ((0, 0), (0, pad_features)), mode='constant', constant_values=0)
        n_features = args.svd_k
    if n_samples < args.svd_k:
        pad_samples = args.svd_k - n_samples
        x = np.pad(x, ((0, pad_samples), (0, 0)), mode='constant', constant_values=0)
        n_samples = args.svd_k
    pca = PCA(n_components=args.svd_k, random_state=args.seed)
    result = torch.tensor(pca.fit_transform(x), dtype=torch.float)
    return result
