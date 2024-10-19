# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
from logging import getLogger
from model.abstract_traffic_state_model import AbstractTrafficStateModel
from model import loss
from scipy.sparse.linalg import eigs
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor,ParameterTuple
from mindspore.common.initializer import initializer, XavierNormal,XavierUniform,Normal, Uniform

def scaled_laplacian(weight):
    """
    compute ~L (scaled laplacian matrix)
    L = D - A
    ~L = 2L/lambda - I

    Args:
        weight(np.ndarray): shape is (N, N), N is the num of vertices

    Returns:
        np.ndarray: ~L, shape (N, N)
    """
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
    """
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Args:
        l_tilde(np.ndarray): scaled Laplacian, shape (N, N)
        k(int): the maximum order of chebyshev polynomials

    Returns:
        list(np.ndarray): cheb_polynomials, length: K, from T_0 to T_{K-1}
    """
    num = l_tilde.shape[0]
    cheb_polynomials = [np.identity(num), l_tilde.copy()]
    for i in range(2, k):
        cheb_polynomials.append(np.matmul(2 * l_tilde, cheb_polynomials[i - 1]) - cheb_polynomials[i - 2])
    return cheb_polynomials



class ChebConv(nn.Cell):
    def __init__(self, k, cheb_polynomials, in_channels, out_channels):
        super(ChebConv, self).__init__()
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

    def construct(self, x):
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = ops.zeros((batch_size, num_of_vertices, self.out_channels), dtype=ms.float32)

            for k in range(self.K):
                t_k = self.cheb_polynomials[k]
                theta_k = self.Theta[k]
                rhs = ops.matmul(self.transpose(graph_signal, (0, 2, 1)), t_k)
                rhs = self.transpose(rhs, (0, 2, 1))
                output = ops.add(output, ops.matmul(rhs, theta_k))

            outputs.append(ops.unsqueeze(output, -1))
        output = ops.concat(outputs, axis=-1)
        output = self.relu(output)
        return output

class MSTGCNBlock(nn.Cell):
    def __init__(self, in_channels, k, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials):
        super(MSTGCNBlock, self).__init__()
        self.cheb_conv = ChebConv(k, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(
            nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
            pad_mode='pad', padding=(0, 0, 1, 1), weight_init=XavierUniform(), bias_init=Uniform()
        )
        self.residual_conv = nn.Conv2d(
            in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides),
            pad_mode='valid', weight_init=XavierUniform(), bias_init=Uniform()
        )
        self.ln = nn.LayerNorm((nb_time_filter,))
        self.relu = nn.ReLU()
        self.transpose = ops.Transpose()

    def construct(self, x):
        spatial_gcn = self.cheb_conv(x)
        spatial_gcn = self.transpose(spatial_gcn, (0, 2, 1, 3))
        time_conv_output = self.time_conv(spatial_gcn)
        x = self.transpose(x, (0, 2, 1, 3))
        x_residual = self.residual_conv(x)
        x_residual = self.relu(x_residual + time_conv_output)
        x_residual = self.transpose(x_residual, (0, 3, 2, 1))
        x_residual = self.ln(x_residual)
        x_residual = self.transpose(x_residual, (0, 2, 3, 1))
        return x_residual


class FusionLayer(nn.Cell):
    # Matrix-based fusion
    def __init__(self, n, h, w):
        super(FusionLayer, self).__init__()
        # define the trainable parameter
        self.weights = Parameter(initializer(XavierUniform(), shape=[1, n, h, w], dtype=ms.float32),
                                 requires_grad=True, name='weights')

    def construct(self, x):
        # assuming x is of size B-n-h-w
        x = x * self.weights  # element-wise multiplication
        return x

class MSTGCNSubmodule(nn.Cell):
    def __init__(self, nb_block, in_channels, k, nb_chev_filter, nb_time_filter,
                 input_window, cheb_polynomials, output_window, output_dim, num_of_vertices):
        super(MSTGCNSubmodule, self).__init__()
        self.block_list = nn.CellList([
            MSTGCNBlock(in_channels, k, nb_chev_filter, nb_time_filter,
                        input_window // output_window, cheb_polynomials)])

        self.block_list.extend([
            MSTGCNBlock(nb_time_filter, k, nb_chev_filter, nb_time_filter, 1, cheb_polynomials)
            for _ in range(nb_block - 1)])

        self.final_conv = nn.Conv2d(in_channels=output_window, out_channels=output_window, #修改了
                                    kernel_size=(1, nb_time_filter - output_dim + 1), has_bias=False,
                                    pad_mode='valid',  # 这里假设不需要padding
                                    weight_init=XavierUniform(),  # 如果需要自定义权重初始化，可以设置这里
                                    bias_init=Uniform()  # 如果需要自定义偏置初始化，可以设置这里
                                  #  weight_init='normal'  # 权重初始化，默认为正态分布
                                    )
        self.output=output_dim


    def construct(self, x):
        """
        Args:
            x: (B, T_in, N_nodes, F_in)

        Returns:
            mindspore.tensor: (B, T_out, N_nodes, out_dim)
        """
        x = ops.Transpose()(x, (0, 2, 3, 1))  # (B, N, F_in(feature_dim), T_in)
        for block in self.block_list:
            x = block(x)
        output = self.final_conv(ops.Transpose()(x, (0, 3, 1, 2)))  # (B, output_window, N_nodes, T_in)

        return output

# 适配最一般的TrafficStateGridDataset和TrafficStatePointDataset
class MSTGCNCommon_model(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.input_dim = self.data_feature.get('input_dim', 1)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
#        self.device = config.get('device', torch.device('cpu'))
        self.nb_block = config.get('nb_block', 2)
        self.K = config.get('K', 3)
        self.nb_chev_filter = config.get('nb_chev_filter', 64)
        self.nb_time_filter = config.get('nb_time_filter', 64)

        adj_mx = self.data_feature.get('adj_mx')
        l_tilde = scaled_laplacian(adj_mx)
        self.cheb_polynomials = [Tensor(i, ms.float32) for i in cheb_polynomial(l_tilde, self.K)]
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        #print("inCommon",self.feature_dim)
        self.MSTGCN_submodule = \
            MSTGCNSubmodule(self.nb_block, self.feature_dim,
                            self.K, self.nb_chev_filter, self.nb_time_filter,
                            self.input_window, self.cheb_polynomials,
                            self.output_window, self.output_dim, self.num_nodes)


    def construct(self, x):
        output = self.MSTGCN_submodule(x)
        return output  # (B, T', N_nodes, F_out)

class MSTGCNCommon(nn.Cell):
    def __init__(self, config, data_feature):
        super(MSTGCNCommon, self).__init__()
        self.loss = loss.masked_mae_m
        self.network = MSTGCNCommon_model(config, data_feature)
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
