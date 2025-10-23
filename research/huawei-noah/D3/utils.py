import random
import numpy as np
import mindspore
import mindspore.nn as nn
import time
import pandas as pd
import gc
import mindspore.dataset as ds
from mindspore import dtype as mstype

class MLP(nn.Cell):
    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Dense(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Dense(input_dim, 1))
        self.mlp = mindspore.nn.SequentialCell(*layers)

    def construct(self, x):
        return self.mlp(x)
    
class channel_attn(nn.Cell):
    def __init__(self, emb) -> None:
        super().__init__()
        self.q_linear = nn.Dense(emb, emb)
        self.k_linear = nn.Dense(emb, emb)
        self.v_linear = nn.Dense(emb, emb)

    def construct(self, x):
        query = self.q_linear(x)
        key = self.k_linear(x)
        value = self.v_linear(x)
        key = key.permute(0, 2, 1)
        attn_weight = mindspore.ops.matmul(query, key) # b,f,f
        attn_weight = mindspore.ops.Softmax(attn_weight, dim=-1)
        out = mindspore.ops.matmul(attn_weight, value) + value # b,f,e
        return out, attn_weight

class DANet(nn.Cell):
    def __init__(self, emb):
        super().__init__()
        self.channel_attn = channel_attn(emb)
    
    def construct(self, x):
        channel_out, channel_attn_weight = self.channel_attn(x)
        out = channel_out
        return out, channel_attn_weight
    
