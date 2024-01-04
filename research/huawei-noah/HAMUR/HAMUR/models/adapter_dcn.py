"""
-*- coding: utf-8 -*-
@Time    : 9/12/2023 8:21 pm
@Author  : Xiaopeng Li
@File    : adapter_dcn.py

"""
import mindspore
import numpy as np
from mindspore import nn, Parameter, Tensor, ParameterTuple, ops
from ..basic.layers import LR, MLP, CrossNetwork, EmbeddingLayer


class DcnMd(nn.Cell):
    """
    Multi-domain Deep & Cross Network adapted for MindSpore.
    """

    def __init__(self, features, num_domains, n_cross_layers, mlp_params):
        super(DcnMd, self).__init__()
        self.features = features
        self.dims = sum([fea.embed_dim for fea in features])
        self.num_domains = num_domains
        self.embedding = EmbeddingLayer(features)
        self.cn = CrossNetwork(self.dims, n_cross_layers)
        self.mlp = MLP(self.dims, output_layer=False, **mlp_params)
        self.linear = LR(self.dims + mlp_params["dims"][-1])

    def construct(self, x):
        domain_id = ops.stop_gradient(x["domain_indicator"].copy())

        embed_x = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # mask list
        mask = []
        # out list
        out = []

        for d in range(self.num_domains):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = embed_x
            cn_out = self.cn(domain_input)
            mlp_out = self.mlp(domain_input)
            x_stack = ops.cat((cn_out, mlp_out), axis=1)
            y = self.linear(x_stack)
            out.append(ops.sigmoid(y))

        final = ops.zeros_like(out[0])
        for d in range(self.num_domains):
            final = ops.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)


class DcnMdAdp(nn.Cell):
    """
    Multi-domain Deep & Cross Network with Adapter adapted for MindSpore.
    """

    def __init__(self, features, num_domains, n_cross_layers, k, mlp_params, hyper_dims):
        super(DcnMdAdp, self).__init__()
        self.features = features
        self.dims = sum([fea.embed_dim for fea in features])
        self.num_domains = num_domains
        self.embedding = EmbeddingLayer(features)
        self.cn = CrossNetwork(self.dims, n_cross_layers)
        self.mlp = MLP(self.dims, output_layer=False, **mlp_params)
        self.linear = LR(self.dims + mlp_params["dims"][-1])
        self.k = k

        u1 = Parameter(Tensor(np.ones((mlp_params["dims"][1], self.k)), mindspore.float32), name="u1")
        u2 = Parameter(Tensor(np.ones((32, self.k)), mindspore.float32), name="u2")

        v1 = Parameter(Tensor(np.ones((self.k, 32)), mindspore.float32), name="v1")
        v2 = Parameter(Tensor(np.ones((self.k, mlp_params["dims"][1])), mindspore.float32), name="v2")

        self.u = ParameterTuple((u1, u2))
        self.v = ParameterTuple((v1, v2))

        # Initialize hyper-network
        hyper_dims += [self.k * self.k]
        input_dim = self.dims
        self.hyper_net = nn.SequentialCell()
        for i_dim in hyper_dims:
            self.hyper_net.append(nn.Dense(input_dim, i_dim))
            self.hyper_net.append(nn.BatchNorm1d(i_dim))
            self.hyper_net.append(nn.ReLU())
            self.hyper_net.append(nn.Dropout(keep_prob=0.99))
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

        emb = self.embedding(x, self.features, squeeze_dim=True)

        mask = []
        out_l = []
        for d in range(self.num_domains):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            # hyper_network_out
            hyper_out_full = self.hyper_net(domain_input)  # B * (k * k)
            # Representation matrix
            hyper_out = hyper_out_full.reshape(-1, self.k, self.k)  # B * k * k

            cn_out = self.cn(domain_input)
            mlp_out = self.mlp(domain_input)

            # ---------
            # # First Adapter-cell
            ein_0 = ops.Einsum('mi,bij,jn->bmn')
            ein_1 = ops.Einsum('bf,bfj->bj')
            # Adapter layer-1: Down projection
            
            w1 = ein_0((self.u[0], hyper_out, self.v[0]))
            b1 = self.b[0]
            tmp_out = ein_1((mlp_out, w1))
            tmp_out += b1
            
            # Adapter layer-2: non-linear
            tmp_out = ops.sigmoid(tmp_out)
            
            # Adapter layer-3: Up - projection
            w2 = ein_0((self.u[1], hyper_out, self.v[1]))
            b2 = self.b[1]
            tmp_out = ein_1((tmp_out, w2))
            tmp_out += b2
            
            # Adapter layer-4: Domain norm
            mean = tmp_out.mean(axis=0)
            var = tmp_out.var(axis=0)
            x_norm = (tmp_out - mean) / ops.sqrt(var + self.eps)
            out = self.gamma1 * x_norm + self.bias1
            # Adapter: short-cut
            mlp_out = out + mlp_out
            # -----------------------

            x_stack = ops.cat([cn_out, mlp_out], axis=1)
            y = self.linear(x_stack)
            out_l.append(ops.sigmoid(y))

        final = ops.zeros_like(out_l[0])
        for d in range(self.num_domains):
            final = ops.where(mask[d].unsqueeze(1), out_l[d], final)

        return final.squeeze(1)
