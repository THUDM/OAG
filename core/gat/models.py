from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from core.gat.layers import BatchMultiHeadGraphAttention


class MatchBatchHGAT(nn.Module):
    def __init__(self, n_type_nodes, n_units=[1433, 8, 7], n_head=8, dropout=0.1,
                 attn_dropout=0.0, instance_normalization=False):
        super(MatchBatchHGAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization

        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)

        d_hidden = n_units[-1]

        self.fc1 = torch.nn.Linear(d_hidden, d_hidden * 3)
        self.fc2 = torch.nn.Linear(d_hidden * 3, d_hidden)
        self.fc3 = torch.nn.Linear(d_hidden, 2)

        self.attentions = BatchMultiHeadGraphAttention(n_head=n_head,
                                                       f_in=n_units[0],
                                                       f_out=n_units[1],
                                                       attn_dropout=attn_dropout,
                                                       n_type_nodes=n_type_nodes)

        self.out_att = BatchMultiHeadGraphAttention(n_head=1,
                                                    f_in=n_head*n_units[1],
                                                    f_out=n_units[2],
                                                    attn_dropout=attn_dropout,
                                                    n_type_nodes=n_type_nodes)

    def forward(self, emb, adj, v_types):
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        bs, n = adj.size()[:2]
        x = self.attentions(emb, adj, v_types)
        x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj, v_types)
        x = F.elu(x)
        x = x.squeeze()

        left_hidden = x[:, 0, :]
        right_hidden = x[:, 1, :]
        v_sim_mul = torch.mul(left_hidden, right_hidden)
        v_sim = self.fc1(v_sim_mul)
        v_sim = F.relu(v_sim)
        v_sim = self.fc2(v_sim)
        v_sim = F.relu(v_sim)
        scores = self.fc3(v_sim)
        return F.log_softmax(scores, dim=1)
