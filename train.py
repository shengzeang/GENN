from __future__ import print_function, division
import random
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from tqdm import tqdm
from utils import *
from model import *
from deepwalk import *
import argparse


set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--embedDim', type=int, default=64, help='Dim of embedding.')
parser.add_argument('--hiddenDim', type=int, default=256, help='Dim of hidden layer.')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--dw', type=bool, default=True, help='use deepwalk or not.')
parser.add_argument('--decoupled', type=bool, default=False, help='decoupled or not.')
args = parser.parse_args()

if args.dataset == 'cora':
    n_input, n_z, n_clusters, name, lr = 1433, args.embedDim, 7, 'cora', args.lr
if args.dataset == 'citeseer':
    n_input, n_z, n_clusters, name, lr = 3703, args.embedDim, 6, 'citeseer', args.lr
if args.dataset == 'wiki':
    n_input, n_z, n_clusters, name, lr = 4973, args.embedDim, 17, 'wiki', args.lr
if args.dataset == 'pubmed':
    n_input, n_z, n_clusters, name, lr = 500, args.embedDim, 3, 'pubmed', args.lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.dw:
    window_size, walk_len, n_walks = 5, 10, 10

if args.dataset == 'wiki':
    adj, data, y, graph = load_wiki()
else:
    adj, data, _, _, _, _, y, graph = load_data(args.dataset)

if args.decoupled:
    model = DNN(args.hiddenDim, n_input=n_input*3, n_z=n_z, dropout=0.).to(device)
else:
    model = GENN(args.hiddenDim, n_input=n_input, n_z=n_z).to(device)
data = data.to(device)
optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-6)

if args.dw:
    embed_dw = deepwalk_train(graph, window_size, walk_len, n_walks, n_z).to(device)

# normalized matrix
adj = adj + sp.eye(adj.shape[0])
adj = normalize(adj)
adj_1 = adj.to_dense()
adj = adj.to(device)

adj_2 = torch.mm(adj_1, adj_1)
adj_3 = torch.mm(adj_2, adj_1)
adj_4 = torch.mm(adj_3, adj_1)

if args.decoupled:
    input_feature = torch.cat((data, torch.mm(adj, data), torch.mm(adj_2.to(device), data)), dim=1).to(device)

kmeans = KMeans(n_clusters=n_clusters, n_init=20)

feature_matrix = exp_matrix(data)
if args.dw:
    topology_s = embedding2similarity(embed_dw)
else:
    topology_s = adj_4.to(device)

gcn_sc = None
sim_loss_list = []
for epoch in range(200 + 1):
    if args.decoupled:
        embed = model(input_feature)
    else:
        embed = model(data, adj)
    if epoch % 5 == 0:
        y_pred = kmeans.fit_predict(embed.data.cpu().numpy())
        eva(y, y_pred, str(epoch))
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

    loss = 1 * fs_loss + 0.01 * ts_loss + 0.01 * (min_dist_loss + 0.1 * max_dist_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
