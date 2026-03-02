import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DualGCN(nn.Module):
    def __init__(self, in1, in2, hidden_dim, out_dim, GCN_num1,GCN_num2):
        super(DualGCN, self).__init__()

        self.gcn1 = GCN(in1, hidden_dim, out_dim, GCN_num1)
        self.gcn2 = GCN(in2, hidden_dim, out_dim, GCN_num2)

    def forward(self, x1, edge_index1, x2, edge_index2):
        out1 = self.gcn1(x1, edge_index1)
        out2 = self.gcn2(x2, edge_index2)
        return out1, out2


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dimension, out_channels, k = 2):
        super(GCN, self).__init__()
        self.layer = nn.ModuleList()
        self.k = k
        self.layer.append(GCNConv(in_channels, hidden_dimension))
        for _ in range(1,k-1):
            self.layer.append(GCNConv(hidden_dimension, hidden_dimension))
        if k > 1:
            self.layer.append(GCNConv(hidden_dimension, out_channels))
        self.act = nn.ReLU()
    def forward(self, x: torch.Tensor,edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.k-1):
            x = self.act(self.layer[i](x, edge_index))
        x = self.layer[-1](x, edge_index)
        return x


def center_away_loss(z):
    z = F.normalize(z)
    z_mean = z.mean(dim=0, keepdim=True)
    distances = torch.norm(z - z_mean, dim=1)
    loss = -distances.mean()
    return loss

def neighbor_close_loss(z,edge_index):
    z = F.normalize(z)
    src, dst = edge_index
    distances = torch.norm(z[src] - z[dst], dim=1)
    loss = distances.mean()
    return loss
