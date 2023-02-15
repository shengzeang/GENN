from __future__ import print_function, division
import random
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import *
from model import *
from deepwalk import *
import argparse

import matplotlib.pyplot as plt
import numpy as np


set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='the value of weight decay.')
parser.add_argument('--k', type=int, default=5, help='the depth of high-order structural information')
parser.add_argument('--embedDim', type=int, default=64, help='Dim of embedding.')
parser.add_argument('--hiddenDim', type=int, default=256, help='Dim of hidden layer.')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--augmented', type=int, default=1, help='use deepwalk or not.')
parser.add_argument('--device', type=int, default=0, help='train on which gpu.')
args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
if args.dataset == 'wiki':
    adj, data, y, graph = load_wiki()
else:
    adj, data, _, _, _, _, y, graph = load_data(args.dataset)

n_node, n_input = data.shape[0], data.shape[1]
n_clusters = (y.max() + 1).item()

if args.augmented:
    print("Using DeepWalk augmentation")
    window_size, walk_len, n_walks = 5, 10, 10
    embed_dw = deepwalk_train(graph, window_size, walk_len, n_walks, args.embedDim).to(device)

model = GENN(args.hiddenDim, n_input=n_input, n_z=args.embedDim).to(device)
data = data.to(device)
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# normalized matrix
adj = adj + sp.eye(adj.shape[0])
adj = normalize(adj)
adj_1 = adj.to_dense()
adj = adj.to(device)

kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)

feature_matrix = exp_matrix(data)
if args.augmented:
    topology_s = embedding2similarity(embed_dw)
else:
    adj_list = [adj_1]
    for _ in range(args.k-1):
        adj_list.append(torch.mm(adj_list[-1], adj_1))
    adj_aggr = adj_1
    for i in range(1, args.k):
        adj_aggr += (1/(i+1)) * adj_list[i]
    topology_s = adj_aggr.to(device)

gcn_sc = None
best_acc = 0.
best_emb = None
best_epoch = 0
sim_loss_list = []
for epoch in range(args.epochs+1):
    embed = model(data, adj)
    if epoch % 5 == 0:
        y_pred = kmeans.fit_predict(embed.data.cpu().numpy())
        acc_cur = eva(y, y_pred, str(epoch))
        if best_acc < acc_cur:
            best_acc = acc_cur
            best_emb = embed
            best_epoch = epoch + 1
        if epoch % 10 == 0:
            centroids = torch.tensor(kmeans.cluster_centers_).to(device)
    maxdist = 0.
    mindist = 0.
    for i in range(n_clusters):
        maxdist += torch.norm(embed - centroids[i], p=2, dim=1).sum().item()
    for i in range(embed.shape[0]):
        mindist += torch.norm(embed[i] - centroids[y_pred[i]], p=2).item()
    min_dist_loss = mindist
    max_dist_loss = mindist / (n_clusters - 1) - maxdist / (n_clusters - 1)

    gcn_sc = embedding2similarity(embed)
    fs_loss = F.kl_div(t_matrix(embed).log(), feature_matrix, reduction='batchmean')
    ts_loss = F.mse_loss(gcn_sc, topology_s)

    loss = 1.0 * fs_loss + 0.01 * ts_loss + 0.01 * (min_dist_loss + 0.1 * max_dist_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
