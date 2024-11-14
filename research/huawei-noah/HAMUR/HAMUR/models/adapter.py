"""
-*- coding: utf-8 -*-
@Time    : 9/12/2023 3:50 pm
@Author  : Xiaopeng Li
@File    : adapter.py

"""
import numpy as np
import mindspore
from mindspore import nn, Parameter, Tensor, ParameterTuple, ops
from ..basic.activation import activation_layer
from ..basic.layers import EmbeddingLayer


class MlpAdap7Layer2Adp(nn.Cell):
    # 7 layers MLP with 2 adapter cells
    def __init__(self, features, domain_num, fcn_dims, hyper_dims, k):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dim = [self.input_dim] + fcn_dims
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)  # Adjust according to MindSpore

        self.relu = activation_layer("relu")  # Make sure activation_layer is compatible with MindSpore
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.CellList()  # MindSpore's CellList as an alternative to PyTorch's ModuleList
        for d in range(self.domain_num):
            domain_specific = nn.SequentialCell()  # MindSpore's SequentialCell
            domain_specific.append(nn.Dense(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[2], self.fcn_dim[3]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[3]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[3], self.fcn_dim[4]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[4]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[4], self.fcn_dim[5]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[5]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[5], self.fcn_dim[6]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[6]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[6], self.fcn_dim[7]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[7]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[7], 1))

            self.layer_list.append(domain_specific)

        # Define parameters for the adapter cells

        self.k = k

        u1 = Parameter(Tensor(np.ones((self.fcn_dim[6], self.k)), mindspore.float32), name="u1")
        u2 = Parameter(Tensor(np.ones((32, self.k)), mindspore.float32), name="u2")
        u3 = Parameter(Tensor(np.ones((self.fcn_dim[7], self.k)), mindspore.float32), name="u3")
        u4 = Parameter(Tensor(np.ones((32, self.k)), mindspore.float32), name="u4")

        v1 = Parameter(Tensor(np.ones((self.k, 32)), mindspore.float32), name="v1")
        v2 = Parameter(Tensor(np.ones((self.k, self.fcn_dim[6])), mindspore.float32), name="v2")
        v3 = Parameter(Tensor(np.ones((self.k, 32)), mindspore.float32), name="v3")
        v4 = Parameter(Tensor(np.ones((self.k, self.fcn_dim[7])), mindspore.float32), name="v4")

        self.u = ParameterTuple((u1, u2, u3, u4))
        self.v = ParameterTuple((v1, v2, v3, v4))

        # hyper-network design
        hyper_dims += [self.k * self.k]
        input_dim = self.input_dim
        self.hyper_net = nn.SequentialCell()
        for i_dim in hyper_dims:
            self.hyper_net.append(nn.Dense(input_dim, i_dim))
            self.hyper_net.append(nn.BatchNorm1d(i_dim))
            self.hyper_net.append(nn.ReLU())
            self.hyper_net.append(nn.Dropout(keep_prob=0.99))
            input_dim = i_dim

        # adapter initiation
        b1 = Parameter(Tensor(np.zeros(32), mindspore.float32), name="b1")
        b2 = Parameter(Tensor(np.zeros((self.fcn_dim[6])), mindspore.float32), name="b2")
        b3 = Parameter(Tensor(np.zeros(32), mindspore.float32), name="b3")
        b4 = Parameter(Tensor(np.zeros((self.fcn_dim[7])), mindspore.float32), name="b4")
        self.b = ParameterTuple((b1, b2, b3, b4))

        self.gamma1 = Parameter(Tensor(np.zeros((self.fcn_dim[6])), mindspore.float32), name="gam1")
        self.bias1 = Parameter(Tensor(np.zeros((self.fcn_dim[6])), mindspore.float32), name="bias1")
        self.gamma2 = Parameter(Tensor(np.zeros((self.fcn_dim[7])), mindspore.float32), name="gam2")
        self.bias2 = Parameter(Tensor(np.zeros((self.fcn_dim[7])), mindspore.float32), name="bias2")
        self.eps = 1e-5

    def construct(self, x):
        domain_id = ops.stop_gradient(x["domain_indicator"].copy())

        emb = self.embedding(x, self.features, squeeze_dim=True)

        mask = []
        out_l = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            # hyper_network_out
            hyper_out_full = self.hyper_net(domain_input)  # B * (k * k)
            # Representation matrix
            hyper_out = hyper_out_full.reshape(-1, self.k, self.k)  # B * k * k

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear

            domain_input = model_list[1](domain_input)  # bn

            domain_input = model_list[2](domain_input)  # relu    B * m

            domain_input = model_list[3](domain_input)  # linear

            domain_input = model_list[4](domain_input)  # bn

            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)  # linear

            domain_input = model_list[7](domain_input)  # bn

            domain_input = model_list[8](domain_input)  # relu

            domain_input = model_list[9](domain_input)  # linear

            domain_input = model_list[10](domain_input)  # bn

            domain_input = model_list[11](domain_input)  # relu

            domain_input = model_list[12](domain_input)  # linear

            domain_input = model_list[13](domain_input)  # bn

            domain_input = model_list[14](domain_input)  # relu

            domain_input = model_list[15](domain_input)  # linear

            domain_input = model_list[16](domain_input)  # bn

            domain_input = model_list[17](domain_input)  # relu

            # # ------------------------------
            # # First Adapter-cell
            ein_0 = ops.Einsum('mi,bij,jn->bmn')
            ein_1 = ops.Einsum('bf,bfj->bj')
            # Adapter layer-1: Down projection

            w1 = ein_0((self.u[0], hyper_out, self.v[0]))
            b1 = self.b[0]
            tmp_out = ein_1(domain_input, w1)
            tmp_out += b1

            # Adapter layer-2: non-linear
            tmp_out = self.sig(tmp_out)

            # Adapter layer-3: Up - projection
            w2 = ein_0(self.u[1], hyper_out, self.v[1])
            b2 = self.b[1]
            tmp_out = ein_1(tmp_out, w2)
            tmp_out += b2

            # Adapter layer-4: Domain norm
            mean = tmp_out.mean(axis=0)
            var = tmp_out.var(axis=0)
            x_norm = (tmp_out - mean) / ops.sqrt(var + self.eps)
            out = self.gamma1 * x_norm + self.bias1

            # Adapter: short-cut
            domain_input = out + domain_input
            # # -----------------------------------

            domain_input = model_list[18](domain_input)  # linear

            domain_input = model_list[19](domain_input)  # bn

            domain_input = model_list[20](domain_input)  # relu

            # # --------------------------------------
            # # Second Adapter-cell
            # # Adapter layer-1: Down projection
            w1 = ein_0(self.u[2], hyper_out, self.v[2])
            b1 = self.b[2]
            tmp_out = ein_1(domain_input, w1)
            tmp_out += b1

            # Adapter layer-2: non-linear
            tmp_out = self.sig(tmp_out)

            # Adapter layer-3: Up - projection
            w2 = ein_0(self.u[3], hyper_out, self.v[3])
            b2 = self.b[3]
            tmp_out = ein_1(tmp_out, w2)
            tmp_out += b2

            # Adapter layer-4: Domain norm
            mean = tmp_out.mean(axis=0)
            var = tmp_out.var(axis=0)
            x_norm = (tmp_out - mean) / ops.sqrt(var + self.eps)
            out = self.gamma2 * x_norm + self.bias2

            # Adapter: short-cut
            domain_input = out + domain_input
            # # --------------------------------------

            domain_input = model_list[21](domain_input)  # linear

            domain_input = self.sig(domain_input)  # relu

            out_l.append(domain_input)

        final = ops.zeros_like(out_l[0])
        for d in range(self.domain_num):
            final = ops.where(mask[d].unsqueeze(1), out_l[d], final)
        return final.squeeze(1)


class MlpAdap2Layer1Adp(nn.Cell):
    # 2 layers MLP with 1 adapter cells
    def __init__(self, features, domain_num, fcn_dims, hyper_dims, k):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dim = [self.input_dim] + fcn_dims
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)  # Adjust according to MindSpore

        self.relu = activation_layer("relu")  # Make sure activation_layer is compatible with MindSpore
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.CellList()  # MindSpore's CellList as an alternative to PyTorch's ModuleList
        for d in range(domain_num):
            domain_specific = nn.SequentialCell()  # MindSpore's SequentialCell
            domain_specific.append(nn.Dense(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[2], 1))

            self.layer_list.append(domain_specific)

        # Define parameters for the adapter cells
        # MindSpore's Parameter initialization
        # instance representation matrix initiation
        self.k = k

        u1 = Parameter(Tensor(np.ones((self.fcn_dim[2], self.k)), mindspore.float32), name="u1")
        u2 = Parameter(Tensor(np.ones((32, self.k)), mindspore.float32), name="u2")

        v1 = Parameter(Tensor(np.ones((self.k, 32)), mindspore.float32), name="v1")
        v2 = Parameter(Tensor(np.ones((self.k, self.fcn_dim[2])), mindspore.float32), name="v2")

        self.u = ParameterTuple((u1, u2))
        self.v = ParameterTuple((v1, v2))

        # hyper-network design
        hyper_dims += [self.k * self.k]
        input_dim = self.input_dim
        self.hyper_net = nn.SequentialCell()
        for i_dim in hyper_dims:
            self.hyper_net.append(nn.Dense(input_dim, i_dim))
            self.hyper_net.append(nn.BatchNorm1d(i_dim))
            self.hyper_net.append(nn.ReLU())
            self.hyper_net.append(nn.Dropout(p=0))
            input_dim = i_dim

        # adapter initiation
        b1 = Parameter(Tensor(np.zeros(32), mindspore.float32), name="b1")
        b2 = Parameter(Tensor(np.zeros((self.fcn_dim[2])), mindspore.float32), name="b2")
        self.b = ParameterTuple((b1, b2))

        self.gamma1 = Parameter(Tensor(np.zeros((self.fcn_dim[2])), mindspore.float32), name="gam1")
        self.bias1 = Parameter(Tensor(np.zeros((self.fcn_dim[2])), mindspore.float32), name="bias1")
        self.eps = 1e-5
        self.ein_0 = ops.Einsum('mi,bij,jn->bmn')
        self.ein_1 = ops.Einsum('bf,bfj->bj')

    def construct(self, x):
        # print("start")
        domain_id = ops.stop_gradient(x["domain_indicator"].copy())

        emb = self.embedding(x, self.features, squeeze_dim=True)

        mask = []
        out_l = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            # hyper_network_out
            hyper_out_full = self.hyper_net(domain_input)  # B * (k * k)
            # Representation matrix
            hyper_out = hyper_out_full.reshape(-1, self.k, self.k)  # B * k * k

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu    B * m

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            # # ----------------------------
            # # First Adapter-cell

            # Adapter layer-1: Down projection
            # print("start2")
            w1 = self.ein_0((self.u[0], hyper_out, self.v[0]))
            # print(w1.shape)
            b1 = self.b[0]
            tmp_out = self.ein_1((domain_input, w1))
            tmp_out += b1

            # Adapter layer-2: non-linear
            tmp_out = self.sig(tmp_out)

            # Adapter layer-3: Up - projection
            w2 = self.ein_0((self.u[1], hyper_out, self.v[1]))
            b2 = self.b[1]
            tmp_out = self.ein_1((tmp_out, w2))
            tmp_out += b2
            # Adapter layer-4: Domain norm
            mean = tmp_out.mean(axis=0)
            var = tmp_out.var(axis=0)
            x_norm = (tmp_out - mean) / ops.sqrt(var + self.eps)
            # print(x_norm)
            out = self.gamma1 * x_norm + self.bias1
            # Adapter: short-cut
            domain_input = out + domain_input
            # ----------------------------

            domain_input = model_list[6](domain_input)  # linear
            domain_input = self.sig(domain_input)  # relu

            out_l.append(domain_input)

        final = ops.zeros_like(out_l[0])
        for d in range(self.domain_num):
            final = ops.where(mask[d].unsqueeze(1), out_l[d], final)
        return final.squeeze(1)


class Mlp2Layer(nn.Cell):
    # 2 layers MLP with 1 adapter cells
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dim = [self.input_dim] + fcn_dims
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)  # Adjust according to MindSpore

        self.relu = activation_layer("relu")  # Make sure activation_layer is compatible with MindSpore
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.CellList()  # MindSpore's CellList as an alternative to PyTorch's ModuleList
        for d in range(domain_num):
            domain_specific = nn.SequentialCell()  # MindSpore's SequentialCell
            domain_specific.append(nn.Dense(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[2], 1))

            self.layer_list.append(domain_specific)

    def construct(self, x):

        domain_id = ops.stop_gradient(x["domain_indicator"].copy())
        emb = self.embedding(x, self.features, squeeze_dim=True)

        mask = []
        out_l = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu    B * m

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)  # linear
            domain_input = self.sig(domain_input)  # relu

            out_l.append(domain_input)

        final = ops.zeros_like(out_l[0])
        for d in range(self.domain_num):
            final = ops.where(mask[d].unsqueeze(1), out_l[d], final)
        return final.squeeze(1)


class Mlp7Layer(nn.Cell):
    # 7 layers MLP with 2 adapter cells
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dim = [self.input_dim] + fcn_dims
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)  # Adjust according to MindSpore

        self.relu = activation_layer("relu")  # Make sure activation_layer is compatible with MindSpore
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.CellList()  # MindSpore's CellList as an alternative to PyTorch's ModuleList
        for d in range(domain_num):
            domain_specific = nn.SequentialCell()  # MindSpore's SequentialCell
            domain_specific.append(nn.Dense(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[2], self.fcn_dim[3]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[3]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[3], self.fcn_dim[4]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[4]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[4], self.fcn_dim[5]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[5]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[5], self.fcn_dim[6]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[6]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[6], self.fcn_dim[7]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[7]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Dense(self.fcn_dim[7], 1))

            self.layer_list.append(domain_specific)

    def construct(self, x):
        domain_id = ops.stop_gradient(x["domain_indicator"].copy())

        emb = self.embedding(x, self.features, squeeze_dim=True)

        mask = []
        out_l = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            # hyper_network_out
            # hyper_out_full = self.hyper_net(domain_input)  # B * (k * k)
            # # Representation matrix
            # hyper_out = hyper_out_full.reshape(-1, self.k, self.k)  # B * k * k

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu    B * m

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)  # linear
            domain_input = model_list[7](domain_input)  # bn
            domain_input = model_list[8](domain_input)  # relu

            domain_input = model_list[9](domain_input)  # linear
            domain_input = model_list[10](domain_input)  # bn
            domain_input = model_list[11](domain_input)  # relu

            domain_input = model_list[12](domain_input)  # linear
            domain_input = model_list[13](domain_input)  # bn
            domain_input = model_list[14](domain_input)  # relu

            domain_input = model_list[15](domain_input)  # linear
            domain_input = model_list[16](domain_input)  # bn
            domain_input = model_list[17](domain_input)  # relu

            domain_input = model_list[18](domain_input)  # linear
            domain_input = model_list[19](domain_input)  # bn
            domain_input = model_list[20](domain_input)  # relu

            domain_input = model_list[21](domain_input)  # linear
            domain_input = self.sig(domain_input)  # relu

            out_l.append(domain_input)

        final = ops.zeros_like(out_l[0])
        for d in range(self.domain_num):
            final = ops.where(mask[d].unsqueeze(1), out_l[d], final)

        return final.squeeze(1)
