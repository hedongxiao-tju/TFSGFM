import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DualGCN(nn.Module):
    def __init__(self, in1, in2, hidden_dim, out_dim, GCN_num1, GCN_num2):
        super(DualGCN, self).__init__()

        self.gcn1 = GCN(in1, hidden_dim, out_dim, GCN_num1)
        self.gcn2 = GCN(in2, hidden_dim, out_dim, GCN_num2)


    def forward(self, x1, edge_index1, x2, edge_index2, args, train = 1):
        if train == 1:
            out1 = self.gcn1(x1, edge_index1)
            out2 = self.gcn2(x2, edge_index2)
            return out1, out2
        else:
            out1 = self.gcn1(x1, edge_index1)
            out2 = self.gcn2(x2, edge_index2)
            z = torch.cat([out1, out2], dim=1)  
            num_hop = args.DAD
            if num_hop != 0:
                a = DAD_edge_index(edge_index1, (z.size()[0], z.size()[0]))
                for i in range(num_hop):
                    z = a @ z
            return z

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


    def forward(self, x: torch.Tensor,edge_index: torch.Tensor, train=1) -> torch.Tensor:
        if train == 1:
            for i in range(self.k-1):
                x = self.act(self.layer[i](x, edge_index))
            x = self.layer[-1](x, edge_index)
            return x
        else:
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

def DAD_edge_index(edge_index, size):
    a = torch.sparse_coo_tensor(edge_index, torch.ones_like(edge_index)[0], size).to_dense().to(
        edge_index.device)
    a = a + torch.eye(n=a.size()[0]).to(edge_index.device)
    d = a.sum(dim=0)
    d_2 = torch.diag(d.pow(-0.5))
    a = d_2 @ a @ d_2
    return a