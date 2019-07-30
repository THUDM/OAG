from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from os.path import join
import torch
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset
import sklearn
import logging

from core.utils import data_utils
from core.utils import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp



class PairedSubgraphDataset(Dataset):

    def __init__(self, file_dir, seed, shuffle):
        self.file_dir = file_dir

        # load subgraphs
        logger.info('loading adjs...')
        self.graphs = np.load(join(file_dir, 'adjacency_matrix.npy'))
        logger.info('adjs loaded')

        # add self-loop
        identity = np.identity(self.graphs.shape[1]).astype(np.bool_)
        self.graphs += identity
        self.graphs[self.graphs != 0] = 1.0
        self.graphs = self.graphs.astype(np.dtype('B'))
        logger.info('graph processed.')
        self.ego_size = self.graphs.shape[1]

        # load node features
        node_to_vec = data_utils.load_vectors(file_dir, 'entity_node_emb.vec')
        logger.info("input node features loaded!")

        # load labels
        self.labels = np.load(os.path.join(file_dir, "label.npy"))
        self.labels = self.labels.astype(np.long)
        logger.info("labels loaded!")

        # load vertices
        self.vertices = np.load(join(file_dir, 'vertex_id.npy'))
        logger.info('vertices loaded')

        # load vertex types
        self.vertex_types = np.load(os.path.join(file_dir, 'vertex_types.npy'))
        logger.info('vertex types loaded')

        if shuffle:
            self.graphs, self.labels, self.vertices, self.vertex_types = \
                sklearn.utils.shuffle(
                self.graphs, self.labels, self.vertices, self.vertex_types,
                random_state=seed
            )

        logger.info('constructing node map...')
        self.all_nodes = set(self.vertices.flatten())
        self.all_nodes_list = list(self.all_nodes)
        self.n_nodes = len(self.all_nodes)
        logger.info('all node count %d', self.n_nodes)
        self.id2idx = {item: i for i, item in enumerate(self.all_nodes_list)}

        self.vertices = np.array(list(map(self.id2idx.get, self.vertices.flatten())),
                                 dtype=np.long).reshape(self.vertices.shape)  # convert to idx

        # order node features
        self.node_feature_dim = len(node_to_vec[list(node_to_vec.keys())[0]])
        logger.info('input node features dim %d', self.node_feature_dim)
        vertex_features = np.zeros((self.n_nodes, self.node_feature_dim))
        n_hit_emb = 0
        for i, eid in enumerate(self.all_nodes_list):
            if i % 10000 == 0:
                logger.info('construct node %d features, n_hit_emb %d', i, n_hit_emb)
            if eid in node_to_vec:
                vertex_features[i] = node_to_vec[eid]
                n_hit_emb += 1
            else:
                vertex_features[i] = np.random.normal(size=(self.node_feature_dim, ))

        self.node_features = torch.FloatTensor(vertex_features)

        self.vertex_types = torch.FloatTensor(self.vertex_types)

        self.N = len(self.graphs)
        logger.info("%d pair ego networks loaded, each with size %d" % (self.N, self.graphs.shape[1]))

    def get_embedding(self):
        return self.node_features

    def get_node_input_feature_dim(self):
        return self.node_feature_dim

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.vertices[idx], self.vertex_types[idx]


if __name__ == '__main__':
    dataset = PairedSubgraphDataset(file_dir=settings.AUTHOR_DATA_DIR, seed=42, shuffle=True)
    
