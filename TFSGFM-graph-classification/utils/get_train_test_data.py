from Dataset_Load import load_graph_classification_dataset
from torch_geometric.utils import to_undirected
from sklearn.decomposition import PCA

import numpy as np
import torch

def get_data(args):
    with (torch.no_grad()):
        train_dataset = []
        for dataset_name in args.train_datasets:
            train_data = load_graph_classification_dataset(dataset_name, args.datasets_dir,args)
            for data in train_data:
                data.x = pca(data.x, args)
                data.x = x_sort(data.x, data.edge_index)
                data.x1, data.x2 = x_split(data.x, args)
                data.edge_index2 = edge_create(data.x2, args, args.top_k)
            train_dataset.append(train_data)

        test_data = load_graph_classification_dataset(args.test_dataset, args.datasets_dir,args)
        for data in test_data:
            data.x = pca(data.x, args)
            data.x = x_sort(data.x, data.edge_index)
            data.x1, data.x2 = x_split(data.x, args)
            data.edge_index2 = edge_create(data.x2, args, args.top_k)
    return train_dataset,test_data

def edge_create(x, args, top_k):
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    sim_matrix = torch.matmul(x_norm, x_norm.t())
    num_nodes = x.size(0)
    topk = top_k
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
    pca = PCA(n_components=args.svd_k, random_state=args.seed)
    result = torch.tensor(pca.fit_transform(x), dtype=torch.float)
    return result[:n_samples]
