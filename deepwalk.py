from __future__ import print_function, division
import random
import numpy as np
import torch
from gensim.models import Word2Vec


def random_walk(G, path_length, rand=random.Random(), start=None):

    walk = [str(start)]
    while len(walk) < path_length:
        cur = int(walk[-1])
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(str(rand.choice(cur_nbrs)))
        else:
            break
    return walk


def par_walk(G, path_length, prob_matrix, rand=random.Random(), start=None):
    walk = [start]
    while len(walk) < path_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            next = rand.choice(cur_nbrs)
            if random.random() > prob_matrix[int(start), int(cur)]:
                walk.append(next)
            else:
                break
        else:
            break
    return walk


def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(G, path_length, rand=rand, start=node))

    return walks

class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, window_size):
        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}
        self.sentences = build_deepwalk_corpus(graph, num_walks, walk_length)
        self.window_size = window_size

    def train(self, n_clusters):
        model = Word2Vec(self.sentences, size=n_clusters, window=self.window_size, sg=1, hs=1)
        self.w2v_model = model
        return model

    def get_embeddings(self):
        if self.w2v_model is None:
            print("model hasn't been trained")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[str(word)] = self.w2v_model.wv[str(word)]
        return self._embeddings

def deepwalk_train(G, window_size, walk_len, n_walks, n_z):
    model_dw = DeepWalk(G, walk_length=walk_len, num_walks=n_walks, window_size=window_size)
    model_dw.train(n_z)
    dict = model_dw.get_embeddings()
    embed = []
    for word in G.nodes():
        embed.append(dict[str(word)])
    embed = torch.Tensor(np.array(embed))

    return embed