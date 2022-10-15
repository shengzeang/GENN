from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output

class GENN(nn.Module):

    def __init__(self, n_enc_1, n_input, n_z):
        super(GENN, self).__init__()

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_z)

    def forward(self, x, adj):
        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2(h, adj)
        return h


class DNN(nn.Module):

    def __init__(self, n_enc, n_input, n_z, dropout=0.):
        super(DNN, self).__init__()
        self.fcn1 = nn.Linear(n_input, n_enc)
        self.fcn2 = nn.Linear(n_enc, n_z)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fcn1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fcn2(x))
        return x