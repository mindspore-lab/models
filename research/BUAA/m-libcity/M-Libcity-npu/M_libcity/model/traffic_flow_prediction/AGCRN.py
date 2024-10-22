from logging import getLogger

import mindspore
from mindspore import nn, ops, numpy
from mindspore.common.initializer import initializer
from model.abstract_model import AbstractModel
import model.loss


class AVWGCN(nn.Cell):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = mindspore.Parameter(initializer('normal', (embed_dim, cheb_k, dim_in, dim_out)), name='weight')
        self.bias_pool = mindspore.Parameter(initializer('Uniform', [embed_dim, dim_out]), name='bias')
        self.softmax = nn.Softmax(axis=1)
        self.relu = nn.ReLU()
        
    def construct(self,x,node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = ops.MatMul(transpose_b=True)(node_embeddings, node_embeddings)
        supports = self.relu(supports)
        supports = self.softmax(supports)
        supports = mindspore.Tensor(supports, dtype=mindspore.float32)
        support_set = [numpy.eye(node_num, dtype=mindspore.float32), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            tmp = mindspore.Tensor(ops.MatMul()(2 * supports, support_set[-1]) - support_set[-2], dtype=mindspore.float32)
            support_set.append(tmp)
        supports = ops.Stack(axis=0)(support_set)

        nshape = node_embeddings.shape
        wshape = self.weights_pool.shape
        node_embeddings_op = node_embeddings.reshape(nshape[0], nshape[1], 1, 1, 1)
        weights_pool_op = self.weights_pool.reshape(1, wshape[0], wshape[1], wshape[2], wshape[3])
        weights = node_embeddings_op * weights_pool_op
        weights = weights.sum(axis=1)
        
        bias = ops.MatMul()(node_embeddings, self.bias_pool)                       # N, dim_out
        x1 = mindspore.Tensor(x, dtype=mindspore.float32)
        supports_op = supports.reshape(1, supports.shape[0], supports.shape[1], supports.shape[2], 1)
        x1_op = x1.reshape(x1.shape[0], 1, 1, x1.shape[1], x1.shape[2])
        x_g = supports_op * x1_op
        x_g = x_g.sum(axis=3)
    
        x_g = ops.Transpose()(x_g, (0, 2, 1, 3))
        # print(x_g.shape)
        
        weights = mindspore.Tensor(weights, dtype=mindspore.float32)
        b,n,k,i=x_g.shape
        n,k,i,o=weights.shape
        x_g=x_g.expand_dims(-1)
        weights=weights.expand_dims(0)
        x_gconv=x_g*weights
        x_gconv=x_gconv.sum(2).sum(2)
        x_gconv = x_gconv + bias     # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Cell):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def construct(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        input_and_state = ops.Concat(axis=-1)((x, state))
        z_r = ops.Sigmoid()(self.gate(input_and_state, node_embeddings))
        z, r = numpy.split(z_r, 2, axis=-1)
        x = mindspore.Tensor(x, dtype=mindspore.float32)
        candidate = ops.Concat(axis=-1)((x, z*state))
        hc = ops.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return numpy.zeros((batch_size, self.node_num, self.hidden_dim), dtype=mindspore.float32)


class AVWDCRNN(nn.Cell):
    def __init__(self, config):
        super(AVWDCRNN, self).__init__()
        self.num_nodes = config['num_nodes']
        self.feature_dim = 1 #config['input_dim']
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 2)
        self.cheb_k = config.get('cheb_order', 3)
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.dcrnn_cells = nn.CellList()
        self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.feature_dim,
                                          self.hidden_dim, self.cheb_k, self.embed_dim))
        for _ in range(1, self.num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim))

    def construct(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.feature_dim
        seq_length = x.shape[1]
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = ops.Stack(axis=1)(inner_states)

        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return ops.Stack(axis=0)(init_states)  # (num_layers, B, N, hidden_dim)


class AGCRN_model(AbstractModel):
    def __init__(self, config, data_feature):
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = 1 #data_feature.get('input_dim', 1)
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim

        super().__init__(config, data_feature)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 12)
        self.output_dim = data_feature.get('output_dim', 1)
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)

        tmp = initializer('xavier_uniform', (self.num_nodes, self.embed_dim), mindspore.float32)
        self.node_embeddings = mindspore.Parameter(tmp, requires_grad=True)

        self.encoder = AVWDCRNN(config)
        self.end_conv = nn.Conv2d(1, self.output_window * self.output_dim, kernel_size=(1, self.hidden_dim), has_bias=True, pad_mode='valid')

        self._logger = getLogger()
        self._scaler = data_feature.get('scaler')
        self.init_parameters_data()

    def construct(self, source):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        # CNN based predictor
        output = self.end_conv(output)                         #B, T*C, N, 1
        output = ops.Squeeze(axis=-1)(output) if output.shape[-1] == 1 else output
        output = ops.Reshape()(output, (-1, self.output_window, self.output_dim, self.num_nodes))
        output = ops.Transpose()(output, (0, 1, 3, 2))  #B, T, N, C
        return output




class AGCRN(nn.Cell):
    def __init__(self, config,data_feature):
        super(AGCRN, self).__init__()
        self.loss = nn.L1Loss()
        self.network = AGCRN_model(config,data_feature)
        self.reshape = ops.Reshape()
        self.mode="train"
        self.zscore = data_feature['scaler']

    def set_loss(self,loss_fn):
        pass

    def train(self):
        self.mode="train"

    def eval(self):
        self.mode="eval"

    def predict(self,x,label):
        y_predict=self.network(x)
        y_predict = self.zscore.inverse_transform(y_predict)
        label = self.zscore.inverse_transform(label)
        return y_predict,label

    def calculate_loss(self,x,label):
        y = self.network(x)
        y = self.zscore.inverse_transform(y)
        label = self.zscore.inverse_transform(label)
        loss = self.loss(y, label)
        return loss

    def construct(self, x, label):
        x=x[...,0:1]
        label=label[...,0:1]
        if self.mode=="train":
            return self.calculate_loss(x,label)
        elif self.mode=="eval":
            return self.predict(x,label)
    
    def evaluate(self, x, label):
        x=x[...,0:1]
        label=label[...,0:1]

        return self.predict(x,label)