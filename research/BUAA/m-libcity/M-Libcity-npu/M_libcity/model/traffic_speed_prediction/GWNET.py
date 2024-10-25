from mindspore import Tensor
import mindspore
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import model.loss
import scipy.sparse as sp
from scipy.sparse import linalg
from model.abstract_traffic_state_model import AbstractTrafficStateModel

class gcn(nn.Cell):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        c_in = (order * support_len + 1) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1)
                             , pad_mode='valid', weight_init='he_uniform')
        self.dropout = dropout
        self.order = order

    def construct(self, x, support):
        out = [x]
        for a in support:
            X = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1)
            A = a.reshape(1, 1, a.shape[0], 1, a.shape[1])
            X_A = X * A
            x1 = X_A.sum(axis=2)
            x1 = x1.transpose(0, 1, 3, 2)
            out.append(x1)
            for k in range(2, self.order + 1):
                X1 = x1.reshape(x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3], 1)
                AA = a.reshape(1, 1, a.shape[0], 1, a.shape[1])
                X_AA = X * AA
                x2 = X_AA.sum(axis=2)
                x2 = x2.transpose(0, 1, 3, 2)
                out.append(x2)
                x1 = x2

        h = ops.Concat(axis=1)(out)
        h = self.mlp(h)
        h, _ = ops.Dropout(self.dropout)(h)
        return h


class gwnet(nn.Cell):
    def __init__(self, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.CellList()
        self.gate_convs = nn.CellList()
        self.residual_convs = nn.CellList()
        self.skip_convs = nn.CellList()
        self.bn = nn.CellList()
        self.gconv = nn.CellList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1), pad_mode='valid', weight_init='he_uniform')
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = mindspore.Parameter(mindspore.Tensor(np.random.randn(num_nodes, 10), mindspore.float32),
                                                    name='nodevec1')
                self.nodevec2 = mindspore.Parameter(mindspore.Tensor(np.random.randn(10, num_nodes), mindspore.float32),
                                                    name='nodevec2')
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = np.linalg.svd(aptinit.to_numpy())
                m, p, n = mindspore.Tensor(m, mindspore.float32), mindspore.Tensor(p,
                                                                                   mindspore.float32), mindspore.Tensor(
                    n, mindspore.float32)
                initemb1 = ops.MatMul()(m[:, :10], mindspore.Tensor(np.diag(p[:10] ** 0.5), mindspore.float32))
                initemb2 = ops.MatMul()(mindspore.Tensor(np.diag(p[:10] ** 0.5), mindspore.float32), n[:, :10].t())
                self.nodevec1 = mindspore.Parameter(initemb1, name='nodevec1')
                self.nodevec2 = mindspore.Parameter(initemb2, name='nodevec1')
                self.supports_len += 1
        count = 0
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation,
                                                   pad_mode='valid', weight_init='he_uniform'))
                self.filter_convs[-1].trainable_params()[0].name = 'filter_convs.' + str(count) + '.weight'
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation, pad_mode='valid',
                                                 weight_init='he_uniform'))
                self.gate_convs[-1].trainable_params()[0].name = 'gate_convs.' + str(count) + '.weight'
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1), pad_mode='valid', weight_init='he_uniform'))
                self.residual_convs[-1].trainable_params()[0].name = 'residual_convs.' + str(count) + '.weight'
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1), pad_mode='valid', weight_init='he_uniform'))
                self.skip_convs[-1].trainable_params()[0].name = 'skip_convs.' + str(count) + '.weight'
                count += 1
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    pad_mode='valid', weight_init='he_uniform')
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    pad_mode='valid', weight_init='he_uniform')

        self.receptive_field = receptive_field

    def construct(self, input):
        in_len = input.shape[3]
        if in_len < self.receptive_field:
            x = ops.Pad(((0, 0), (0, 0), (0, 0), (self.receptive_field - in_len, 0)))(input)
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = ops.Softmax(axis=1)(ops.ReLU()(ops.MatMul()(self.nodevec1, self.nodevec2)))
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            residual = x
            filter = self.filter_convs[i](residual)
            filter = ops.Tanh()(filter)
            gate = self.gate_convs[i](residual)
            gate = ops.Sigmoid()(gate)
            x = ops.Mul()(filter, gate)
            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            if not isinstance(skip, int):
                skip = skip[:, :, :, -s.shape[3]:]
            else:
                skip = 0
            skip = s + skip
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.shape[3]:]
            x = self.bn[i](x)
        x = ops.ReLU()(skip)
        x = ops.ReLU()(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

class GWNET(nn.Cell):
    def __init__(self, config, data_feature):
        super(GWNET, self).__init__()
        self.loss = nn.L1Loss()
        self.zscore = data_feature['scaler']

        adj_mx = data_feature['adj_mx']
        adjtype = config['adjtype']
        adj_mx = self.load_adj(adjtype, adj_mx)
        supports = [mindspore.Tensor(i, mindspore.float32) for i in adj_mx]
        dropout = config['dropout']
        gcn_bool = config['gcn_bool']
        addaptadj = config['addaptadj']
        feature_dim = 1#data_feature['feature_dim']
        output_dim = 1#data_feature['output_dim']
        residual_channels = config['residual_channels']
        dilation_channels = config['dilation_channels']
        skip_channels = config['skip_channels']
        end_channels = config['end_channels']
        node_num = data_feature['num_nodes']
        print('!', dropout, gcn_bool, addaptadj, feature_dim, output_dim, residual_channels, dilation_channels, skip_channels,
              end_channels, node_num)
        self.network = gwnet(node_num, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=None,
                    in_dim=feature_dim, out_dim=12, residual_channels=residual_channels,
                    dilation_channels=dilation_channels, skip_channels=skip_channels,
                    end_channels=end_channels)
        self.reshape = ops.Reshape()
        self.output_window = config.get('output_window', 12)
        self.output_dim = 1#config.get('output_dim', 1)
        self.mode = "train"

    def load_adj(self, adjtype, adj_mx):
        if adjtype == "scalap":
            adj = [self.calculate_scaled_laplacian(adj_mx)]
        elif adjtype == "normlap":
            adj = [self.calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            adj = [self.sym_adj(adj_mx)]
        elif adjtype == "transition":
            adj = [self.asym_adj(adj_mx)]
        elif adjtype == "doubletransition":
            adj = [self.asym_adj(adj_mx), self.asym_adj(np.transpose(adj_mx))]
        elif adjtype == "identity":
            adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
        else:
            adj = 0
            error = 0
            assert error, "adj type not defined"
        return adj

    def calculate_normalized_laplacian(self, adj):
        """
        # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
        # D = diag(A 1)
        :param adj:
        :return:
        """
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian

    def calculate_scaled_laplacian(self, adj_mx, lambda_max=2, undirected=True):
        if undirected:
            adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
        L = self.calculate_normalized_laplacian(adj_mx)
        if lambda_max is None:
            lambda_max, _ = linalg.eigsh(L, 1, which='LM')
            lambda_max = lambda_max[0]
        L = sp.csr_matrix(L)
        M, _ = L.shape
        I = sp.identity(M, format='csr', dtype=L.dtype)
        L = (2 / lambda_max * L) - I
        return L.astype(np.float32).todense()

    def asym_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        return d_mat.dot(adj).astype(np.float32).todense()

    def sym_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


    def set_loss(self, loss_fn):
        pass

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"
        
    def predict(self, x, label):
        x = x.transpose(0, 3, 2, 1)
        x = ops.Pad(((0, 0), (0, 0), (0, 0), (1, 0)))(x)
        y = self.network(x)
        y = self.zscore.inverse_transform(y)
        label = self.zscore.inverse_transform(label)
        return y, label

    def calculate_loss(self, x, label):
        
        x = x.transpose(0, 3, 2, 1)
        x = ops.Pad(((0, 0), (0, 0), (0, 0), (1, 0)))(x)
        y = self.network(x)
        y = self.zscore.inverse_transform(y)
        label = self.zscore.inverse_transform(label)
        loss = self.loss(y, label)
        return loss

    def construct(self, x, label):
        x = x.astype(dtype=mindspore.float32)
        x = x[:,:,:,0:1]
        label = label[:,:,:,0:1]
        if self.mode == "train":
            return self.calculate_loss(x, label)
        elif self.mode == "eval":
            return self.predict(x, label)
    
    def evaluate(self, x, label):
        x = x.astype(dtype=mindspore.float32)
        x = x[:,:,:,0:1]
        label = label[:,:,:,0:1]

        return self.predict(x, label)