"""
-*- coding: utf-8 -*-
@Time    : 9/12/2023 8:21 pm
@Author  : Xiaopeng Li
@File    : adapter_wd.py

"""
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, Parameter, ops, ParameterTuple
import numpy as np
from ..basic.layers import LR, MLP, EmbeddingLayer


class WideDeepMd(nn.Cell):
    """
    Multi-domain Wide & Deep Learning model adapted for MindSpore.
    """

    def __init__(self, wide_features, num_domains, deep_features, mlp_params):
        super(WideDeepMd, self).__init__()
        self.num_domains = num_domains
        self.wide_features = wide_features
        self.deep_features = deep_features
        self.wide_dims = sum([fea.embed_dim for fea in wide_features])
        self.deep_dims = sum([fea.embed_dim for fea in deep_features])

        self.linear = LR(self.wide_dims)
        self.embedding = EmbeddingLayer(wide_features + deep_features)
        self.mlp = MLP(self.deep_dims, **mlp_params)

    def construct(self, x):
        domain_id = ops.stop_gradient(x["domain_indicator"].copy())

        input_wide = self.embedding(x, self.wide_features, squeeze_dim=True)  # [batch_size, wide_dims]
        input_deep = self.embedding(x, self.deep_features, squeeze_dim=True)  # [batch_size, deep_dims]

        mask = []
        out = []

        for d in range(self.num_domains):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input_wide = input_wide
            domain_input_deep = input_deep

            y_wide = self.linear(domain_input_wide)  # [batch_size, 1]
            y_deep = self.mlp(domain_input_deep)  # [batch_size, 1]
            y = y_wide + y_deep
            out.append(ops.sigmoid(y))

        final_output = ops.zeros_like(out[0])
        for d in range(self.num_domains):
            final_output = ops.where(mask[d].unsqueeze(1), out[d], final_output)

        return final_output.squeeze(1)


class WideDeepMdAdp(nn.Cell):
    def __init__(self, wide_features, num_domains, deep_features, k, mlp_params, hyper_dims):
        super(WideDeepMdAdp, self).__init__()
        self.num_domains = num_domains
        self.wide_features = wide_features
        self.deep_features = deep_features
        self.wide_dims = sum([fea.embed_dim for fea in wide_features])
        self.deep_dims = sum([fea.embed_dim for fea in deep_features])
        self.linear = LR(self.wide_dims)
        self.embedding = EmbeddingLayer(wide_features + deep_features)
        self.mlp = MLP(self.deep_dims, **mlp_params, output_layer=False)
        self.mlp_final = LR(mlp_params["dims"][-1])

        self.k = k

        u1 = Parameter(Tensor(np.ones((mlp_params["dims"][-1], self.k)), mindspore.float32), name="u1")
        u2 = Parameter(Tensor(np.ones((32, self.k)), mindspore.float32), name="u2")

        v1 = Parameter(Tensor(np.ones((self.k, 32)), mindspore.float32), name="v1")
        v2 = Parameter(Tensor(np.ones((self.k, mlp_params["dims"][-1])), mindspore.float32), name="v2")

        self.u = ParameterTuple((u1, u2))
        self.v = ParameterTuple((v1, v2))

        # Initialize hyper-network
        hyper_dims += [self.k * self.k]
        input_dim = self.wide_dims + self.deep_dims
        self.hyper_net = nn.SequentialCell()
        for i_dim in hyper_dims:
            self.hyper_net.append(nn.Dense(input_dim, i_dim))
            self.hyper_net.append(nn.BatchNorm1d(i_dim))
            self.hyper_net.append(nn.ReLU())
            self.hyper_net.append(nn.Dropout(keep_prob=0.99-mlp_params["dropout"]))
            input_dim = i_dim

        # adapter initiation
        b1 = Parameter(Tensor(np.zeros((32,)), mindspore.float32), name="b1")
        b2 = Parameter(Tensor(np.zeros(mlp_params["dims"][-1]), mindspore.float32), name="b2")
        self.b = ParameterTuple((b1, b2))

        self.gamma1 = Parameter(Tensor(np.ones(mlp_params["dims"][-1]), mindspore.float32))
        self.bias1 = Parameter(Tensor(np.zeros(mlp_params["dims"][-1]), mindspore.float32))
        self.eps = 1e-5

    def construct(self, x):
        domain_id = ops.stop_gradient(x["domain_indicator"].copy())

        input_wide = self.embedding(x, self.wide_features, squeeze_dim=True)  # [batch_size, wide_dims]
        input_deep = self.embedding(x, self.deep_features, squeeze_dim=True)  # [batch_size, deep_dims]

        hyper_out_full = self.hyper_net(ops.concat((input_wide, input_deep), axis=1))  # B * (k * k)
        hyper_out = hyper_out_full.reshape(-1, self.k, self.k)  # B * k * k

        # Mask and output storage
        mask = []
        out_l = []

        for d in range(self.num_domains):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input_wide = input_wide
            domain_input_deep = input_deep

            y_wide = self.linear(domain_input_wide)  # [batch_size, 1]
            y_deep = self.mlp(domain_input_deep)  # [batch_size, f]

            # ---------------------------------
            # First Adapter-cell
            ein_0 = ops.Einsum('mi,bij,jn->bmn')
            ein_1 = ops.Einsum('bf,bfj->bj')
            # Adapter layer-1: Down projection
            
            w1 = ein_0((self.u[0], hyper_out, self.v[0]))
            b1 = self.b[0]
            tmp_out = ein_1((y_deep, w1))
            tmp_out += b1
            
            # Adapter layer-2: non-linear
            tmp_out = ops.sigmoid(tmp_out)
            # Adapter layer-3: Up - projection
            w2 = ein_0((self.u[1], hyper_out, self.v[1]))
            b2 = self.b[1]
            tmp_out = ein_1((tmp_out, w2))
            tmp_out += b2
            
            # adapter layer-4: Domain norm
            mean = tmp_out.mean(axis=0)
            var = tmp_out.var(axis=0)
            x_norm = (tmp_out - mean) / ops.sqrt(var + self.eps)
            out = self.gamma1 * x_norm + self.bias1
            
            # Adapter: short-cut
            mlp_out = out + y_deep
            # # ---------------------------------

            mlp_out = self.mlp_final(mlp_out)
#             mlp_out = self.mlp_final(y_deep)  # linear

            y = y_wide + mlp_out

            out_l.append(ops.sigmoid(y))

        final = ops.zeros_like(out_l[0])
        for d in range(self.num_domains):
            final = ops.where(mask[d].unsqueeze(1), out_l[d], final)
        return final.squeeze(1)
