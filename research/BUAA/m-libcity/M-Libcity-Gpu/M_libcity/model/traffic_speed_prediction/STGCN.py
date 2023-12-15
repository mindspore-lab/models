import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer
import numpy as np
import model.loss
import model.utils
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigs

class Align(nn.Cell):
    """
    # align channel_in and channel_out
    """

    def __init__(self, channel_in, channel_out):
        super(Align, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.align_conv = nn.Conv2d(in_channels=self.channel_in, out_channels=self.channel_out, kernel_size=1,
                                    pad_mode='valid', weight_init='he_uniform')

    def construct(self, x):
        x_align = x
        if self.channel_in > self.channel_out:
            x_align = self.align_conv(x)
        elif self.channel_in < self.channel_out:
            dim1, dim2, dim3, dim4 = x.shape
            y = ops.Zeros()((dim1, self.channel_out - self.channel_in, dim3, dim4), x.dtype)
            x_align = ops.Concat(axis=1)((x, y))
        return x_align


class TemporalConvLayer(nn.Cell):
    """
    # 1.align
    # 2.conv2d
    # 3.glu or gtu
    """

    def __init__(self, t_kernel_size, channel_in, channel_out, vertex_num, gate_type):
        super(TemporalConvLayer, self).__init__()
        self.t_kernel_size = t_kernel_size
        self.gate_type = gate_type
        self.align = Align(channel_in, channel_out)
        self.conv2d = nn.Conv2d(channel_in, 2 * channel_out, (t_kernel_size, 1), stride=(1, 1),
                                padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True,
                                weight_init='he_uniform')

    def construct(self, x):
        x_in = self.align(x)
        x_in = x_in[:, :, self.t_kernel_size - 1:, :]
        x_conv2d = self.conv2d(x)
        x_pq = ops.Split(axis=1, output_num=2)(x_conv2d)
        if self.gate_type == 'glu':  # glu
            # (x_p + x_in) ⊙ Sigmoid(x_q)
            x_glu = ops.Mul()(ops.Add()(x_pq[0], x_in), nn.Sigmoid()(x_pq[1]))
            x_tc_out = x_glu
        else:  # self.gate_type == 'gtu'
            # Tanh(x_p + x_in) ⊙ Sigmoid(x_q)
            x_gtu = ops.Mul()(nn.Tanh()(ops.Add()(x_pq[0], x_in)), nn.Sigmoid()(x_pq[1]))
            x_tc_out = x_gtu
        return x_tc_out


class ChebConv(nn.Cell):
    """
    # cheb
    """

    def __init__(self, channel_in, channel_out, cheb_k, chebconv_matrix):
        super(ChebConv, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.cheb_k = cheb_k
        self.chebconv_matrix = chebconv_matrix
        self.weight = mindspore.Parameter(initializer('normal', (self.cheb_k, channel_in, self.channel_out)),
                                          name='weight')
        self.bias = mindspore.Parameter(initializer('Uniform', [self.channel_out]), name='bias')

    def construct(self, x):
        _, channel_in, _, vertex_num = x.shape
        # K = cheb_k - 1
        x = ops.Reshape()(x, (vertex_num, -1))
        x_0 = x
        x_1 = ops.MatMul()(self.chebconv_matrix, x)
        x_list = []
        if self.cheb_k - 1 == 0:
            x_list = [x_0]
        elif self.cheb_k - 1 == 1:
            x_list = [x_0, x_1]
        elif self.cheb_k - 1 >= 2:
            x_list = [x_0, x_1]
            for k in range(2, self.cheb_k):
                x_list.append(ops.MatMul()(2 * self.chebconv_matrix, x_list[k - 1]) - x_list[k - 2])
        x_tensor = ops.Stack(axis=0)(x_list)
        x_mul = ops.MatMul()(ops.Reshape()(x_tensor, (-1, self.cheb_k * channel_in)), ops.Reshape()(self.weight,
                                                                                                    (
                                                                                                        self.cheb_k * channel_in,
                                                                                                        -1)))
        x_mul = ops.Reshape()(x_mul, (-1, self.channel_out))
        x_chebconv = ops.BiasAdd()(x_mul, self.bias)
        return x_chebconv


class GCNConv(nn.Cell):
    """
    # GCN
    """

    def __init__(self, channel_in, channel_out, gcnconv_matrix):
        super(GCNConv, self).__init__()
        self.channel_out = channel_out
        self.gcnconv_matrix = gcnconv_matrix
        self.weight = mindspore.Parameter(initializer('he_uniform', (channel_in, channel_out)), name='weight')
        self.bias = mindspore.Parameter(initializer('Uniform', [self.channel_out]), name='bias')

    def construct(self, x):
        """gcnconv compute"""
        _, channel_in, _, vertex_num = x.shape
        x_first_mul = ops.MatMul()(ops.Reshape()(x, (-1, channel_in)), self.weight)
        x_first_mul = ops.Reshape()(x_first_mul, (vertex_num, -1))
        x_second_mul = ops.MatMul()(self.gcnconv_matrix, x_first_mul)
        x_second_mul = ops.Reshape()(x_second_mul, (-1, self.channel_out))
        x_gcnconv_out = x_second_mul
        return x_gcnconv_out


class GraphConvLayer(nn.Cell):
    """
    # 1.align
    # 2.chebconv or gcnconv
    """

    def __init__(self, cheb_k, channel_in, channel_out, graph_conv_type, graph_conv_matrix):
        super(GraphConvLayer, self).__init__()
        self.channel_out = channel_out
        self.align = Align(channel_in, channel_out)
        self.graph_conv_type = graph_conv_type
        if self.graph_conv_type == "chebconv":
            self.chebconv = ChebConv(self.channel_out, self.channel_out, cheb_k, graph_conv_matrix)
        elif self.graph_conv_type == "gcnconv":
            self.gcnconv = GCNConv(self.channel_out, self.channel_out, graph_conv_matrix)

    def construct(self, x):
        """GraphConvLayer compute"""
        x_gc_in = self.align(x)
        batch_size, _, T, vertex_num = x_gc_in.shape
        x_gc = x_gc_in
        if self.graph_conv_type == "chebconv":
            x_gc = self.chebconv(x_gc_in)
        elif self.graph_conv_type == "gcnconv":
            x_gc = self.gcnconv(x_gc_in)
        x_gc_with_rc = ops.Add()(ops.Reshape()(x_gc, (batch_size, self.channel_out, T, vertex_num)), x_gc_in)
        x_gc_out = x_gc_with_rc
        return x_gc_out


class STConvBlock(nn.Cell):
    """
    # 1.TemporalConvLayer
    # 2.GraphConvLayer
    # 3.TemporalConvLayer
    # 4.Normalization
    # 5.Dropout
    """

    def __init__(self, t_kernel_size, cheb_k, vertex_num, last_block_channel, channels, gate_type, graph_conv_type,
                 graph_conv_matrix, drop_rate):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(t_kernel_size, last_block_channel, channels[0],
                                           vertex_num, gate_type)
        self.graph_conv = GraphConvLayer(cheb_k, channels[0], channels[1],
                                         graph_conv_type, graph_conv_matrix)
        self.tmp_conv2 = TemporalConvLayer(t_kernel_size, channels[1], channels[2],
                                           vertex_num, gate_type)
        self.tc2_ln = nn.LayerNorm([vertex_num, channels[2]], begin_norm_axis=2,
                                   begin_params_axis=2, epsilon=1e-05)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(keep_prob=drop_rate)

    def construct(self, x):
        x_tmp_conv1 = self.tmp_conv1(x)
        x_graph_conv = self.graph_conv(x_tmp_conv1)
        x_graph_conv_relu = self.relu(x_graph_conv)
        x_tmp_conv2 = self.tmp_conv2(x_graph_conv_relu)
        x_tc2_ln = ops.Transpose()(x_tmp_conv2, (0, 2, 3, 1))
        x_tc2_ln = self.tc2_ln(x_tc2_ln)
        x_tc2_ln = ops.Transpose()(x_tc2_ln, (0, 3, 1, 2))
        x_do = self.do(x_tc2_ln)
        return x_do


class OutputBlock(nn.Cell):
    """
    # 1.TemporalConvLayer
    # 2.LayerNorm
    # 3.fully-connected
    # 4.fully-connected
    """

    def __init__(self, output_t_kernel_size, last_block_channel, channels, end_channel, vertex_num, gate_type):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(output_t_kernel_size, last_block_channel,
                                           channels[0], vertex_num, gate_type)
        self.t_layerNorm = nn.LayerNorm([vertex_num, channels[0]], begin_norm_axis=2,
                                        begin_params_axis=2, epsilon=1e-05)
        self.fc1 = nn.Dense(channels[0], channels[1]).to_float(mindspore.float16)
        self.fc2 = nn.Dense(channels[1], end_channel).to_float(mindspore.float16)

        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        """OutputBlock compute"""
        x_t = self.tmp_conv1(x)
        x_t_ln = self.t_layerNorm(ops.Transpose()(x_t, (0, 2, 3, 1)))
        x_fc1 = self.fc1(x_t_ln)
        x_fc1_sigmoid = self.sigmoid(x_fc1)
        x_fc2 = ops.Transpose()(self.fc2(x_fc1_sigmoid), (0, 3, 1, 2))
        x_out = x_fc2
        return x_out


class _STGCN(nn.Cell):
    """
    # 1.STConvBlock
    # 2.STConvBlock
    # 3.OutputBlock
    """

    def __init__(self, t_kernel_size, cheb_k, blocks, data_t, vertex_num, gate_type, graph_conv_type, chebconv_matrix,
                 drop_rate):
        super(_STGCN, self).__init__()
        modules = []
        """
        # contains 2 STConvBlock
        """
        for i in range(len(blocks) - 3):
            modules.append(STConvBlock(t_kernel_size, cheb_k, vertex_num, blocks[i][-1], blocks[i + 1],
                                       gate_type, graph_conv_type, chebconv_matrix, drop_rate))
        self.st_blocks = nn.SequentialCell(modules)

        output_t_kernel_size = data_t - (len(blocks) - 3) * 2 * (t_kernel_size - 1)
        self.output = OutputBlock(output_t_kernel_size, blocks[-3][-1], blocks[-2],
                                  blocks[-1][0], vertex_num, gate_type)

    def construct(self, x):
        x_stbs = self.st_blocks(x)
        x_out = self.output(x_stbs)
        return x_out


class STGCN(nn.Cell):
    def __init__(self, config, data_feature):
        super(STGCN, self).__init__()
        self.loss = model.loss.masked_rmse_m
        self.zscore = data_feature['scaler']
        blocks = config['blocks']
        t_kernel_size = config['Kt']
        cheb_k = 2
        n_his = config['input_window']
        node_num = data_feature['num_nodes']
        gated_act_func = "glu"
        graph_conv_type = config['graph_conv_type']
        adj_mx = data_feature['adj_mx']
        drop_rate = 0.5
        n_pred = config['n_pred']
        if graph_conv_type == "chebconv":
            if n_pred > 3:
                mat_type = 1
            else:
                mat_type = 2
        else:
            if n_pred > 3:
                mat_type = 3
            else:
                mat_type = 4
        self.n_pred = n_pred
        conv_matrix = self.calculate_laplacian_matrix(adj_mx, mat_type)
        print(t_kernel_size, cheb_k, blocks, n_his, node_num,
                        gated_act_func, graph_conv_type, conv_matrix, drop_rate)
        self.network = _STGCN(t_kernel_size, cheb_k, blocks, n_his, node_num,
                        gated_act_func, graph_conv_type, conv_matrix, drop_rate)
        self.reshape = ops.Reshape()
        self.output_window = config.get('output_window', 12)
        self.output_dim = config.get('output_dim', 1)
        self.mode = "train"

    def calculate_laplacian_matrix(self, adj_mat, mat_type):
        n_vertex = adj_mat.shape[0]

        # row sum
        deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
        # column sum
        deg_mat = deg_mat_row

        adj_mat = np.asmatrix(adj_mat)
        id_mat = np.asmatrix(np.identity(n_vertex))

        # Combinatorial
        com_lap_mat = deg_mat - adj_mat

        # For SpectraConv
        sym_normd_lap_mat = np.matmul(np.matmul(fractional_matrix_power(deg_mat, -0.5), \
                                                com_lap_mat), fractional_matrix_power(deg_mat, -0.5))

        # For ChebConv
        lambda_max_sym = eigs(sym_normd_lap_mat, k=1, which='LR')[0][0].real
        wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / lambda_max_sym - id_mat

        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_sym_normd_lap_mat = np.matmul(np.matmul(fractional_matrix_power(wid_deg_mat, -0.5), \
                                                    wid_adj_mat), fractional_matrix_power(wid_deg_mat, -0.5))

        # Random Walk
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)

        # For SpectraConv
        rw_normd_lap_mat = id_mat - rw_lap_mat

        # For ChebConv
        # From [0, 1] to [-1, 1]
        lambda_max_rw = eigs(rw_lap_mat, k=1, which='LR')[0][0].real
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat

        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)

        if mat_type == 1:
            return mindspore.Tensor(mindspore.Tensor.from_numpy(wid_sym_normd_lap_mat), mindspore.float32)
        if mat_type == 2:
            return mindspore.Tensor(mindspore.Tensor.from_numpy(wid_rw_normd_lap_mat), mindspore.float32)
        if mat_type == 3:
            return mindspore.Tensor(mindspore.Tensor.from_numpy(hat_sym_normd_lap_mat), mindspore.float32)
        if mat_type == 4:
            return mindspore.Tensor(mindspore.Tensor.from_numpy(hat_rw_normd_lap_mat), mindspore.float32)
        raise ValueError('Unknown Type')


    def set_loss(self, loss_fn):
        pass

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"

    def predict(self, x, label):
        x = x[:, :, :, 0:1].transpose(0, 3, 1, 2)
        label = label[:, 0:1, :, 0:1]
        x = self.network(x)
        x = self.zscore.inverse_transform(x)
        label = self.zscore.inverse_transform(label)
        x = x.transpose(0, 1, 3, 2)
        return x, label
    
    def forward(self, batch):
        x = batch['X']
        x = x[:, :, :, 0].expand_dims(axis=-1).transpose(0, 3, 1, 2)
        x = self.network(x)
        x = self.reshape(x, (len(x), -1))
        # x = self.zscore.inverse_transform(x)
        x = x.expand_dims(axis=1)
        x = x.expand_dims(axis=-1)
        return x
    
    def multi_predict(self, x, y):
        # 多步预测
        y_preds = []
        x_numpy = x[..., 0].expand_dims(axis=-1).asnumpy()
        x_ = Tensor.from_numpy(x_numpy)
        for i in range(self.output_window):
            batch_tmp = {'X': x_}
            y_ = self.forward(batch_tmp)  # (batch_size, 1, num_nodes, output_dim)
            y__numpy = y_.asnumpy()
            y__clone = Tensor.from_numpy(y__numpy)
            y_preds.append(y__clone)
            if y_.shape[-1] < x_.shape[-1]:  # output_dim < feature_dim
                y_ = ops.Concat(axis=3)([y_.astype(mindspore.float32), y[:, i:i+1, :, self.output_dim:]])
            x_ = ops.Concat(axis=1)([x_[:, 1:, :, :], y_.astype(mindspore.float32)])
        y_preds = ops.Concat(axis=1)(y_preds)  # (batch_size, output_length, num_nodes, output_dim)
        return self.zscore.inverse_transform(y_preds[..., 0]), self.zscore.inverse_transform(y[..., 0])

    def calculate_loss(self, x, label):
        x = x[:, :, :, 0].expand_dims(axis=-1).transpose(0, 3, 1, 2)
        label = label[:, 0, :, 0]
        x = self.network(x)
        x = self.reshape(x, (len(x), -1))
        label = self.reshape(label, (len(label), -1))
        x = self.zscore.inverse_transform(x)
        label = self.zscore.inverse_transform(label)
        loss = self.loss(x, label)
        return loss

    def construct(self, x, label):
        x = x.astype(dtype=mindspore.float32)
        if self.mode == "train":
            return self.calculate_loss(x, label)
        elif self.mode == "eval":
            return self.multi_predict(x, label)
