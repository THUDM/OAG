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
from torch.utils.data.sampler import Sampler
import sklearn
import logging
from utils import data_utils
from utils import settings

logger = logging.getLogger(__name__)


class ChunkSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class PairedLocalGraphDataset(Dataset):
    edge_index = None

    def __init__(self, file_dir, embedding_dim, seed, shuffle):
        self.file_dir = file_dir
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

        df = pd.read_csv(join(file_dir, 'line.emb'), sep=' ', encoding='us-ascii', header=None)
        line_emb_dict = dict()
        line_i = 0
        for index, row in df.iterrows():
            if line_i % 10000 == 0:
                logger.info('line %d', line_i)
            row = row.tolist()
            line_i += 1
            cur_emb = row[1:1+embedding_dim]
            assert not np.isnan(cur_emb).any()
            line_emb_dict[str(row[0])] = [float(s) for s in cur_emb]
        logger.info('line emb loaded')

        self.labels = np.load(os.path.join(file_dir, "label.npy"))
        # self.labels = np.load(os.path.join(file_dir, "labels_correct.npy"))
        self.labels = self.labels.astype(np.long)
        logger.info("labels loaded!")

        logger.info('loading vertices...')
        self.vertices = np.load(join(file_dir, 'vertex_id.npy'))
        logger.info('vertices loaded')

        self.vertex_types = np.load(os.path.join(file_dir, 'vertex_types.npy'))
        logger.info('vertex types loaded')

        self.svm_features = np.load(join(file_dir, 'features_4_harden_tentative_harden_True.npy'))
        logger.info('svm features loaded')
        self.dim_pair_features = self.svm_features.shape[1]

        if shuffle:
            self.graphs, self.labels, self.vertices, self.vertex_types, self.svm_features = \
            sklearn.utils.shuffle(
                self.graphs, self.labels, self.vertices, self.vertex_types, self.svm_features,
                random_state=seed
            )

        logger.info('constructing node map...')
        self.all_nodes = set(self.vertices.flatten())
        # self.all_nodes = set(itertools.chain.from_iterable(v for v in self.vertices))
        self.all_nodes_list = list(self.all_nodes)
        self.n_nodes = len(self.all_nodes)
        logger.info('all node count %d', self.n_nodes)
        self.id2idx = {item: i for i, item in enumerate(self.all_nodes_list)}

        self.vertices = np.array(list(map(self.id2idx.get, self.vertices.flatten())), dtype=np.long).reshape(
            self.vertices.shape)  # convert to idx

        logger.info('loading node content embedding...')
        emb_dict = data_utils.load_data(file_dir, 'entity_emb_dict.pkl')
        logger.info('node content embedding loaded')
        vertex_features = np.zeros((self.n_nodes, embedding_dim))
        line_emb_mat = np.zeros((self.n_nodes, embedding_dim))
        n_miss = 0
        n_miss_line = 0
        for i, eid in enumerate(self.all_nodes_list):
            if i % 10000 == 0:
                logger.info('node idx %d, n_miss %d, n_miss_line %d', i, n_miss, n_miss_line)
            if eid in emb_dict:
                vertex_features[i] = emb_dict[eid]
            else:
                n_miss += 1
                vertex_features[i] = np.random.normal(size=(embedding_dim, ))
            if eid in line_emb_dict:
                line_emb_mat[i] = line_emb_dict[eid]
            else:
                n_miss_line += 1
                line_emb_mat[i] = np.random.normal(size=(embedding_dim, ))

        vertex_features = torch.FloatTensor(vertex_features)
        self.node_features = vertex_features
        self.line_emb_mat = line_emb_mat

        self.vertex_types = torch.FloatTensor(self.vertex_types)
        self.svm_features = torch.FloatTensor(self.svm_features)

        self.N = len(self.graphs)
        logger.info("%d pair ego networks loaded, each with size %d" % (self.N, self.graphs.shape[1]))

    def get_embedding(self):
        return self.node_features

    def get_ego_size(self):
        return self.ego_size

    def get_line_emb_mat(self):
        return self.line_emb_mat

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.vertices[idx], \
               self.vertex_types[idx], self.svm_features[idx]


class PairwiseStatFeatures(Dataset):
    def __init__(self, file_dir, seed, shuffle):
        self.svm_features = np.load(join(file_dir, 'features_4_harden_tentative_harden_True.npy')).astype(np.float32)
        logger.info('svm features loaded')
        self.dim_pair_features = self.svm_features.shape[1]

        self.labels = np.load(os.path.join(file_dir, "label.npy"))
        self.labels = self.labels.astype(np.float32)
        logger.info("labels loaded!")

        if shuffle:
            self.labels, self.svm_features = \
            sklearn.utils.shuffle(
                self.labels, self.svm_features,
                random_state=seed
            )

        self.N = len(self.labels)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.labels[idx], self.svm_features[idx]


def processing_batch(i_batch, batch, embs):
    graphs, labels, vertices, vertex_types, svm_features = batch
    graphs = graphs.numpy()
    bs = len(graphs)
    v_flatten = vertices.numpy().flatten()
    batch_features = torch.FloatTensor(embs[v_flatten])
    v_types_flatten = torch.LongTensor(vertex_types.numpy().flatten())
    coo_mats = []
    n_ego_size = graphs.shape[1]
    for i in range(bs):
        rows, cols = np.nonzero(graphs[i])
        nnz = len(rows)
        cur_sp_mat = coo_matrix((np.ones(nnz), (rows, cols)), shape=(n_ego_size, n_ego_size))
        coo_mats.append(cur_sp_mat)
    sp_diag_block = sp.block_diag(coo_mats)

    indices = np.vstack((sp_diag_block.row, sp_diag_block.col))
    i = torch.LongTensor(indices)
    values = sp_diag_block.data
    v = torch.FloatTensor(values)
    shape = sp_diag_block.shape
    adj_torch = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    labels = labels.long()
    if i_batch % 10 == 0:
        logger.info('batch %d processed', i_batch)
    return adj_torch, labels, batch_features, v_types_flatten, svm_features


def dense_graph_to_sp_torch_tensor(graphs):
    bs = len(graphs)
    n_ego_size = graphs.shape[1]
    coo_mats = []
    for i in range(bs):
        r = np.nonzero(graphs[i])
        rows, cols = r[:, 0], r[:, 1]
        nnz = len(rows)
        cur_sp_mat = coo_matrix((np.ones(nnz), (rows, cols)), shape=(n_ego_size, n_ego_size))
        coo_mats.append(cur_sp_mat)
    sp_diag_block = sp.block_diag(coo_mats)

    indices = np.vstack((sp_diag_block.row, sp_diag_block.col))
    i = torch.LongTensor(indices)
    values = sp_diag_block.data
    v = torch.FloatTensor(values)
    shape = sp_diag_block.shape
    adj_torch = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return adj_torch

