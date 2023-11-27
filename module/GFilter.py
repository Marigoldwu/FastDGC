# -*- coding: utf-8 -*-
"""
@Time: 2023/10/21 12:53 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GFilter(Module):
    def __init__(self, in_features, out_features):
        super(GFilter, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, times):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if times > 1:
            for i in range(times-1):
                output = torch.spmm(adj, output)
        return output
