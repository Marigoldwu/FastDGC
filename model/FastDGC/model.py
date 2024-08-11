# -*- coding: utf-8 -*-
"""
@Time: 2023/6/8 15:12 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
from module.EGAE import EGAE


class FastDGC(nn.Module):
    def __init__(self, input_dim, linear_dim, embedding_dim, clusters, v=1):
        super(FastDGC, self).__init__()
        self.egae = EGAE(input_dim, linear_dim, embedding_dim)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(clusters, embedding_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v

    def forward(self, x, adj_norm, times):
        A_pred, embedding, x_d = self.egae(x, adj_norm, times)

        q = 1.0 / (1.0 + torch.sum(torch.pow(embedding.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return A_pred, embedding, x_d, q


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred
