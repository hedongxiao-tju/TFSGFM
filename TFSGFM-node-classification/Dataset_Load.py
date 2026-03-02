import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork


def load_dataset(dataset_name, dataset_dir, args):
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name=dataset_name,
                            transform=T.NormalizeFeatures())
    elif dataset_name in ['Cornell']:
        return WebKB(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())
    elif dataset_name in ['Chameleon', 'Squirrel']:
        return WikipediaNetwork(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())
    return dataset




