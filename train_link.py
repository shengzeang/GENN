from __future__ import print_function, division
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import *
from model import *
from deepwalk import *
import argparse


set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='the value of weight decay.')
parser.add_argument('--embedDim', type=int, default=64, help='Dim of embedding.')
parser.add_argument('--hiddenDim', type=int, default=256, help='Dim of hidden layer.')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--simplified', action='store_true', help='disentangled or not.')
parser.add_argument('--device', type=int, default=0, help='train on which gpu.')
args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

if args.dataset == 'wiki':
    adj, data, y, graph = load_wiki()
else:
    adj, data, _, _, _, _, y, graph = load_data(args.dataset)

n_input = data.shape[1]
n_clusters = (y.max() + 1).item()

window_size, walk_len, n_walks = 5, 10, 10
embed_dw = deepwalk_train(graph, window_size, walk_len, n_walks, args.embedDim).to(device)

adj_orig = adj
adj_orig = adj_orig - \
        sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [
                      0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, _, _, _, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

if args.simplified:
    model = DNN(args.hiddenDim, n_input=n_input*3, n_z=args.embedDim).to(device)
else:
    model = GENN(args.hiddenDim, n_input=n_input, n_z=args.embedDim).to(device)
data = data.to(device)
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# normalized matrix
adj = adj + sp.eye(adj.shape[0])
adj = normalize(adj)
adj_1 = adj.to_dense()
adj = adj.to(device)

adj_2 = torch.mm(adj_1, adj_1)
adj_3 = torch.mm(adj_2, adj_1)
adj_4 = torch.mm(adj_3, adj_1)

if args.simplified:
    input_feature = torch.cat((data, torch.mm(adj, data), torch.mm(adj_2.to(device), data)), dim=1).to(device)

kmeans = KMeans(n_clusters=n_clusters, n_init=20)

feature_matrix = exp_matrix(data).to(device)
topology_s = embedding2similarity(embed_dw)

gcn_sc = None
sim_loss_list = []
for epoch in range(400 + 1):
    if args.simplified:
        embed = model(input_feature)
    else:
        embed = model(data, adj)
    if epoch % 5 == 0:
        sim = torch.sigmoid(torch.mm(embed.data.cpu(), embed.data.cpu().T))
        roc_score, ap_score = get_roc_score(sim.numpy(), adj_orig, test_edges, test_edges_false)
        print(f'Epoch {epoch}: AUC: {roc_score:.4f}, AP: {ap_score:.4f}')
        if epoch % 10 == 0:
            y_pred = kmeans.fit_predict(embed.data.cpu().numpy())
            centroids = torch.tensor(kmeans.cluster_centers_).to(device)
    maxdist = 0.
    mindist = 0.
    for i in range(n_clusters):
        maxdist += torch.norm(embed - centroids[i], p=2, dim=1).sum().item()
    for i in range(embed.shape[0]):
        mindist += torch.norm(embed[i] - centroids[y_pred[i]], p=2).item()
    min_dist_loss = mindist / data.shape[0]
    max_dist_loss = (mindist / (n_clusters - 1) - maxdist / (n_clusters - 1)) / data.shape[0]

    gcn_sc = embedding2similarity(embed)
    fs_loss = F.kl_div(t_matrix(embed).log(), feature_matrix, reduction='batchmean')
    ts_loss = F.mse_loss(gcn_sc, topology_s)

    loss = 1 * fs_loss + 1 * ts_loss + 0.01 * (min_dist_loss + 0.1 * max_dist_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
