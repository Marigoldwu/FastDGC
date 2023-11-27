# -*- coding: utf-8 -*-
"""
@Time: 2023/6/25 10:51 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from module.GFilter import GFilter


class EGAE(nn.Module):
    def __init__(self, input_dim, linear_dim, embedding_dim):
        super(EGAE, self).__init__()
        self.fcn = Linear(input_dim, linear_dim)
        self.gcn = GFilter(linear_dim, embedding_dim)

    def forward(self, x, adj_norm, times):
        xt = self.fcn(x)
        h = self.gcn(xt, adj_norm, times)
        embedding = F.normalize(h, p=2, dim=1)
        A_pred = dot_product_decode(embedding)
        return A_pred, embedding, xt


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred
