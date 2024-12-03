from logging import getLogger

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Parameter, Tensor, ParameterTuple
from mindspore.common.initializer import initializer, XavierUniform, Uniform
from model import loss
from model.abstract_traffic_state_model import AbstractTrafficStateModel
from scipy.sparse.linalg import eigs


def scaled_laplacian(weight):
    assert weight.shape[0] == weight.shape[1]
    n = weight.shape[0]
    diag = np.diag(np.sum(weight, axis=1))
    lap = diag - weight
    for i in range(n):
        for j in range(n):
            if diag[i, i] > 0 and diag[j, j] > 0:
                lap[i, j] /= np.sqrt(diag[i, i] * diag[j, j])
    lambda_max = eigs(lap, k=1, which='LR')[0].real
    return (2 * lap) / lambda_max - np.identity(weight.shape[0])


def cheb_polynomial(l_tilde, k):
    num = l_tilde.shape[0]
    cheb_polynomials = [np.identity(num), l_tilde.copy()]
    for i in range(2, k):
        cheb_polynomials.append(np.matmul(2 * l_tilde, cheb_polynomials[i - 1]) - cheb_polynomials[i - 2])
    return cheb_polynomials



class SpatialAttentionLayer(nn.Cell):

    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(SpatialAttentionLayer, self).__init__()
        self.W1 = Parameter(initializer(Uniform(), shape=[num_of_timesteps], dtype=ms.float32))
        self.W2 = Parameter(initializer(XavierUniform(), shape=[in_channels, num_of_timesteps], dtype=ms.float32))
        self.W3 = Parameter(initializer(Uniform(), shape=[in_channels], dtype=ms.float32))
        self.bs = Parameter(initializer(XavierUniform(), shape=[1, num_of_vertices, num_of_vertices], dtype=ms.float32))
        self.Vs = Parameter(initializer(XavierUniform(), shape=[num_of_vertices, num_of_vertices], dtype=ms.float32))
        self.transpose=ops.Transpose()

    def construct(self, x):
        lhs = ops.matmul(x, self.W1)
        lhs = ops.matmul(lhs, self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = ops.matmul(self.W3, x)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        rhs = self.transpose(rhs,(0,2,1))
        product = ops.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        s = ops.matmul(self.Vs, ops.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        s_normalized = ops.softmax(s, axis=1)

        return s_normalized


class ChebConvWithSAt(nn.Cell):
    def __init__(self, k, cheb_polynomials, in_channels, out_channels):
        super(ChebConvWithSAt, self).__init__()
        self.K = k
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = ParameterTuple(
            [Parameter(initializer(XavierUniform(), shape=[in_channels, out_channels], dtype=ms.float32),
                       requires_grad=True, name=f'param_{i}') for i in range(k)]
        )
        self.transpose = ops.Transpose()
        self.relu = nn.ReLU()

    def construct(self, x,spatial_attention):
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = ops.zeros((batch_size, num_of_vertices, self.out_channels), dtype=ms.float32)
            for k in range(self.K):
                t_k = self.cheb_polynomials[k]
                t_k_with_at = ops.mul(t_k, spatial_attention)
                theta_k = self.Theta[k]
                t_k_with_at = self.transpose(t_k_with_at, (0, 2, 1))
                rhs = ops.matmul(t_k_with_at,graph_signal)
                output = output+ops.matmul(rhs, theta_k)
            outputs.append(ops.unsqueeze(output, -1))
        output = ops.concat(outputs, axis=-1)
        output = self.relu(output)
        return output



class TemporalAttentionLayer(nn.Cell):
    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(TemporalAttentionLayer, self).__init__()
        self.U1 = Parameter(initializer(Uniform(), shape=[num_of_vertices], dtype=ms.float32))
        self.U2 = Parameter(initializer(XavierUniform(), shape=[in_channels, num_of_vertices], dtype=ms.float32))
        self.U3 = Parameter(initializer(Uniform(), shape=[in_channels], dtype=ms.float32))
        self.be = Parameter(initializer(XavierUniform(), shape=[1, num_of_timesteps, num_of_timesteps], dtype=ms.float32))
        self.Ve = Parameter(initializer(XavierUniform(), shape=[num_of_timesteps, num_of_timesteps], dtype=ms.float32))
        self.transpose = ops.Transpose()

    def construct(self, x):

        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = ops.matmul(ops.matmul(self.transpose(x,(0, 3, 2, 1)), self.U1), self.U2)

        rhs = ops.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = ops.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        e = ops.matmul(self.Ve, ops.sigmoid(product + self.be))  # (B, T, T)

        e_normalized = ops.softmax(e, axis=1)

        return e_normalized


class ASTGCNBlock(nn.Cell):
    def __init__(self, in_channels, k, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(ASTGCNBlock, self).__init__()
        self.TAt = TemporalAttentionLayer( in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = SpatialAttentionLayer(in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = ChebConvWithSAt(k, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(
            nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
            pad_mode='pad', padding=(0, 0, 1, 1),
        )
        self.residual_conv = nn.Conv2d(
            in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides),
            pad_mode='valid',
        )
        self.ln = nn.LayerNorm((nb_time_filter,))
        self.transpose=ops.Transpose()
        self.relu=nn.ReLU()

    def construct(self, x):
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # TAt
        temporal_at = self.TAt(x)  # (B, T, T)

        x_tat = ops.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_at)\
            .reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        # (B, N*F_in, T) * (B, T, T) -> (B, N*F_in, T) -> (B, N, F_in, T)

        # SAt
        spatial_at = self.SAt(x_tat)  # (B, N, N)

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_at)  # (B, N, F_out, T), F_out = nb_chev_filter

        # convolution along the time axis
        time_conv_output = self.time_conv(self.transpose(spatial_gcn,(0, 2, 1, 3)))
        # (B, N, F_out, T) -> (B, F_out, N, T) 用(1,3)的卷积核去做->(B, F_out', N, T') F_out'=nb_time_filter

        # residual shortcut
        x_residual = self.residual_conv(self.transpose(x,(0, 2, 1, 3)))
        # (B, N, F_in, T) -> (B, F_in, N, T) 用(1,1)的卷积核去做->(B, F_out', N, T') F_out'=nb_time_filter

        x_residual = self.relu(x_residual + time_conv_output)
        x_residual = self.transpose(x_residual,(0, 3, 2, 1))
        x_residual = self.ln(x_residual)
        x_residual = self.transpose(x_residual, (0, 2, 3, 1))
        # (B, F_out', N, T') -> (B, T', N, F_out') -ln -> (B, T', N, F_out') -> (B, N, F_out', T')

        return x_residual


class ASTGCNSubmodule(nn.Cell):
    def __init__(self,nb_block, in_channels, k, nb_chev_filter, nb_time_filter,
                 input_window, cheb_polynomials, output_window, output_dim, num_of_vertices):
        super(ASTGCNSubmodule, self).__init__()

        self.BlockList = nn.CellList([ASTGCNBlock(in_channels, k, nb_chev_filter,
                                                    nb_time_filter, input_window // output_window,
                                                    cheb_polynomials, num_of_vertices, input_window)])

        self.BlockList.extend([ASTGCNBlock(nb_time_filter, k, nb_chev_filter,
                                           nb_time_filter, 1, cheb_polynomials,
                                           num_of_vertices, output_window)
                               for _ in range(nb_block-1)])


        self.final_conv = nn.Conv2d(in_channels=output_window, out_channels=output_window,
                                    kernel_size=(1, nb_time_filter - output_dim + 1), has_bias=False,
                                    pad_mode='valid',
                                    )

        self.transpose = ops.Transpose()

    def construct(self, x):
        x = self.transpose(x,(0, 2, 3, 1))  # (B, N, F_in(feature_dim), T_in)
        for block in self.BlockList:
            x = block(x)
        # (B, N, F_out(nb_time_filter), T_out(output_window))
        output = self.final_conv(self.transpose(x,(0, 3, 1, 2)))
        # (B,N,F_out,T_out)->(B,T_out,N,F_out)-conv<1,F_out-out_dim+1>->(B,T_out,N,out_dim)
        return output

class ASTGCNCommon_model(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.nb_block = config.get('nb_block', 2)
        self.K = config.get('K', 3)
        self.nb_chev_filter = config.get('nb_chev_filter', 64)
        self.nb_time_filter = config.get('nb_time_filter', 64)

        adj_mx = self.data_feature.get('adj_mx')
        l_tilde = scaled_laplacian(adj_mx)
        self.cheb_polynomials = [Tensor(i, ms.float32) for i in cheb_polynomial(l_tilde, self.K)]
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        self.ASTGCN_submodule = \
            ASTGCNSubmodule( self.nb_block, self.feature_dim,
                            self.K, self.nb_chev_filter, self.nb_time_filter,
                            self.input_window, self.cheb_polynomials,
                            self.output_window, self.output_dim, self.num_nodes)

    def construct(self, x):
        output = self.ASTGCN_submodule(x)
        return output

class ASTGCNCommon(nn.Cell):
    def __init__(self, config, data_feature):
        super(ASTGCNCommon, self).__init__()
        self.loss = loss.masked_mae_m
        self.network = ASTGCNCommon_model(config, data_feature)
        self.mode = "train"
        self.zscore = data_feature['scaler']
        self.transpose = ops.Transpose()
        self.output_dim = data_feature['output_dim']

    def train(self):
        self.mode = "train"
        self.set_grad(True)
        self.set_train(True)

    def eval(self):
        self.mode = "eval"
        self.set_grad(False)
        self.set_train(False)

    def validate(self):
        self.set_grad(False)
        self.set_train(False)

    def calculate_loss(self, x,label):
        y = self.network(x)
        y = self.zscore.inverse_transform(y)#[..., :self.output_dim]
        label = self.zscore.inverse_transform(label)#[..., :self.output_dim]
        return loss.masked_mse_m(y, label,0)

    def predict(self, x,label):
        y_predict = self.network(x)
        y_predict = self.zscore.inverse_transform(y_predict)#[..., :self.output_dim]
        label = self.zscore.inverse_transform(label)#[..., :self.output_dim]
        return y_predict,label

    def construct(self, x, label):
        if self.mode=="train":
            return self.calculate_loss(x,label)
        elif self.mode=="eval":
            return self.predict(x,label)

