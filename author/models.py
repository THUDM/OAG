from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from author.gat_layers import BatchMultiHeadGraphAttention


class MatchBatchGAT(nn.Module):
    def __init__(self, pretrained_emb, batch_size, vertex_feature, use_vertex_feature,
                 n_type_nodes,dim_pair_feature, alpha, n_units=[1433, 8, 7], n_head=8,
                 dropout=0.1, attn_dropout=0.0, fine_tune=False, instance_normalization=False, cuda=False):
        super(MatchBatchGAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.batch_size = batch_size
        self.dropout = dropout
        self.inst_norm = instance_normalization

        self.use_vertex_feature = use_vertex_feature
        if self.use_vertex_feature:
            n_units[0] += pretrained_emb.size(1)
            pass

        if self.inst_norm:
            # self.norm = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        d_o1 = n_units[-1]
        # n_concat = dim_pair_feature + d_o1
        n_concat = d_o1

        self.fc1 = torch.nn.Linear(n_concat, n_concat * 3)
        self.fc2 = torch.nn.Linear(n_concat * 3, n_concat)
        self.fc3 = torch.nn.Linear(n_concat, 2)

        self.attentions = BatchMultiHeadGraphAttention(n_head=n_head,
                                                       f_in=n_units[0],
                                                       f_out=n_units[1],
                                                       attn_dropout=attn_dropout,
                                                       n_type_nodes=n_type_nodes)
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        self.out_att = BatchMultiHeadGraphAttention(n_head=1,
                                                    f_in=n_head*n_units[1],
                                                    f_out=n_units[2],
                                                    attn_dropout=attn_dropout,
                                                    n_type_nodes=n_type_nodes)

    def forward(self, features, adj, bs, num_v, svm_features, line_emb, v_types):
        # emb = features.unsqueeze(0)
        emb = torch.cat((features, line_emb), dim=2)
        # emb = features
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        # x = torch.squeeze(emb, 0)
        bs, n = adj.size()[:2]
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x, attn1 = self.attentions(emb, adj, v_types)
        x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
        x = F.dropout(x, self.dropout, training=self.training)
        x, _ = self.out_att(x, adj, v_types)
        x = F.elu(x)
        # x = self.out_att(x, adj)
        x = x.squeeze()

        # left_idx = [i * num_v for i in range(bs)]
        # right_idx = [i * num_v + 1 for i in range(bs)]
        # left_hidden = x[left_idx]
        # right_hidden = x[right_idx]
        left_hidden = x[:, 0, :]
        right_hidden = x[:, 1, :]

        v_sim_mul = torch.mul(left_hidden, right_hidden)
        # v_sim_cat = torch.cat((svm_features, v_sim_mul), dim=1)
        v_sim_cat = v_sim_mul
        v_sim = self.fc1(v_sim_cat)
        v_sim = F.relu(v_sim)
        v_sim = self.fc2(v_sim)
        v_sim = F.relu(v_sim)
        scores = self.fc3(v_sim)

        return F.log_softmax(scores, dim=1), v_sim_mul, attn1
