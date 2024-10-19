import math
import mindspore
from logging import getLogger
from mindspore import nn
from mindspore import  ops
from mindspore.common import dtype as mstype
from mindspore.nn.loss import SmoothL1Loss
from mindspore.ops import einsum
import numpy as np
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from model.abstract_model import AbstractModel
from model import loss
from mindspore.common.initializer import initializer, XavierNormal,XavierUniform,Normal, Uniform
from mindspore import context
# 设置运行模式和设备
context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
class gcn_operation(nn.Cell):
    def __init__(self, adj, in_dim, out_dim, num_vertices, activation='GLU'):
        super(gcn_operation, self).__init__()
        self.adj = mindspore.Tensor(adj, dtype=mindspore.float32)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation
        assert self.activation in {'GLU', 'relu'}

        # 创建权重和偏置的Parameter对象
        self.weight = Parameter(initializer(XavierUniform(),shape=(out_dim, in_dim),
            dtype=mindspore.float32),requires_grad=True)#.to('cuda')

        self.bias = Parameter(initializer(Uniform(),shape=(out_dim,),
            dtype=mindspore.float32),requires_grad=True)#.to('cuda')

        if self.activation == 'GLU':
            self.fc = nn.Dense(in_dim, 2 * out_dim,has_bias=True)
        else:
            self.fc = nn.Dense(in_dim, out_dim,has_bias=True)

        self.relu = ops.ReLU()
        self.sigmoid = ops.Sigmoid()
        self.split = ops.Split(axis=-1, output_num=2)
        self.matmul = ops.MatMul()
    def construct(self, x, mask=None):
        """
        定义前向传播函数。
        :param x: 输入特征张量，其形状为(3*N, B, Cin)，其中N是节点数，B是批次大小，Cin是输入通道数。
        :param mask: 掩码张量，用于屏蔽邻接矩阵中不需要更新的部分，其形状为(3*N, 3*N)。
        :return: 输出特征张量，其形状为(3*N, B, Cout)。
        """
        if mask is not None:
            adj = self.adj * mask
        else:
            adj = self.adj

        x_reshaped = x.reshape(x.shape[0], -1)  # 形状变为 (num_vertices, batch_size * channels_in)
        result = self.matmul(adj, x_reshaped)  # 结果形状为 (num_vertices, batch_size * channels_in)
        x = result.reshape(adj.shape[0], -1, self.in_dim)  # 结果形状为 (num_vertices, batch_size, channels_in)
        x = self.fc(x)

        if self.activation == 'GLU':
            # 分割后，lhs和rhs的形状都为3*N, B, Cout
            lhs, rhs = self.split(x)
            # 应用GLU激活函数，即门控机制
            out = lhs * self.sigmoid(rhs)
            return out
        elif self.activation == 'relu':
            return self.relu(x)


class STSGCM(nn.Cell):
    def __init__(self, adj, in_dim, out_dims, num_of_vertices, activation='GLU'):
        super(STSGCM, self).__init__()

        self.adj = Tensor(adj, dtype=mstype.float32)
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.gcn_operations = nn.CellList()

        self.gcn_operations.append(
            gcn_operation(
                adj=self.adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation
            )
        )

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_operation(
                    adj=self.adj,
                    in_dim=self.out_dims[i - 1],
                    out_dim=self.out_dims[i],
                    num_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )


    def construct(self, x, mask=None):
        need_concat = []
        for i in range(len(self.out_dims)):
            if mask is not None:
                x = self.gcn_operations[i](x, mask)
            else:
                x = self.gcn_operations[i](x)

            need_concat.append(x)

        need_concat = [
            ops.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
            ) for h in need_concat
        ]


        out=ops.Concat(axis=0)(need_concat)

        out,index = ops.max(out,axis=0)

        return out

class STSGCL(nn.Cell):
    def __init__(self,
                 adj,
                 history,
                 num_of_vertices,
                 in_dim,
                 out_dims,
                 strides=3,
                 activation='GLU',
                 temporal_emb=True,
                 spatial_emb=True):
        super(STSGCL, self).__init__()
        self.adj = adj
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        # STFGNN的扩张卷积  1d
        self.dilation_conv_1 = nn.Conv2d(
            self.in_dim,
            self.in_dim,
            kernel_size=(1, 2),
            stride=(1, 1),
            dilation=(1, 3),
            has_bias=True,  # 包含偏置项
            pad_mode='valid',   # 不进行填充
            padding=0,
            weight_init=XavierUniform(),
            bias_init=Uniform()
        )

        self.dilation_conv_2 = nn.Conv2d(
            self.in_dim,
            self.in_dim,
            kernel_size=(1, 2),
            stride=(1, 1),
            dilation=(1, 3),
            has_bias=True,  # 包含偏置项
            pad_mode='valid',  # 不进行填充
            padding=0,
            weight_init=XavierUniform(),
            bias_init=Uniform()
        )

        self.STSGCMS = nn.CellList()
        for i in range(self.history - self.strides + 1):
            self.STSGCMS.append(
                STSGCM(
                    adj=self.adj,
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )
        # 时间嵌入参数的初始化
        if temporal_emb:
            temporal_embedding_init = XavierUniform()
            temporal_embedding_param = Parameter(
                Tensor(shape=[1, history, 1, in_dim], dtype=mstype.float32, init=temporal_embedding_init)
            ,requires_grad=True)
            self.insert_param_to_cell('temporal_embedding', temporal_embedding_param)

        # 空间嵌入参数的初始化
        if spatial_emb:
            spatial_embedding_init = XavierUniform()
            spatial_embedding_param = Parameter(
                Tensor(shape=[1, 1, num_of_vertices, in_dim], dtype=mstype.float32, init=spatial_embedding_init)
            ,requires_grad=True)
            self.insert_param_to_cell('spatial_embedding', spatial_embedding_param)

            self._get_temporal_embedding_shape=[1, history, 1, in_dim]
            self._get_spatial_embedding_shape=[1, 1, num_of_vertices, in_dim]
            self.reset()

    def reset(self):
        # 如果temporal_emb为True，则重新初始化时间嵌入参数
        if self.temporal_emb:
            self.temporal_embedding = Parameter(initializer(
                XavierNormal(gain=0.0003),
                shape=self._get_temporal_embedding_shape,
                dtype=mindspore.float32,)
            ,requires_grad=True)

        # 如果spatial_emb为True，则重新初始化空间嵌入参数
        if self.spatial_emb:
            self.spatial_embedding = Parameter(initializer(
                XavierNormal(gain=0.0003),
                shape=self._get_spatial_embedding_shape,
                dtype=mindspore.float32
            )
            ,requires_grad=True)

    def construct(self, x, mask=None):
        if self.temporal_emb:
            x = x + self.temporal_embedding
        if self.spatial_emb:
            x = x + self.spatial_embedding

        # STFGNN版本
        x_temp = ops.Transpose()(x, (0, 3, 2, 1))  # (B, Cin, N, T)
        x_left = ops.Tanh()(self.dilation_conv_1(x_temp))
        x_right = ops.Sigmoid()(self.dilation_conv_2(x_temp))

        x_time_axis = x_left * x_right
        x_res = ops.Transpose()(x_time_axis, (0, 3, 2, 1))  # (B, N, T-3, C)

        need_concat = []
        batch_size = x.shape[0]

        for i in range(self.history - self.strides + 1):
            # (B, 3, N, Cin) -> (B, 3*N, Cin)
            t = x[:, i: i + self.strides, :, :]
            t = ops.Reshape()(t, (batch_size, self.strides * self.num_of_vertices, self.in_dim))

            # (3*N, B, Cin) -> (N, B, Cout)
            t = self.STSGCMS[i](ops.Transpose()(t, (1, 0, 2)), mask)

            # (N, B, Cout) -> (B, 1, N, Cout)
            t = ops.Transpose()(t, (1, 0, 2))
            t = ops.unsqueeze(t, dim=1)

            need_concat.append(t)


        out = ops.Concat(axis=1)(need_concat)  # (B, T-2, N, Cout*(history-strides+1))
        layer_out = out + x_res
        return layer_out

class output_layer(nn.Cell):
    def __init__(self, num_of_vertices, history, in_dim,hidden_dim=128, horizon=12):
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.history = history
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.fc1 = nn.Dense(
            in_channels=self.in_dim * history,
            out_channels=self.hidden_dim,
            has_bias=True
        )
        self.fc2 = nn.Dense(
            in_channels=self.hidden_dim,
            out_channels=self.horizon,
            has_bias=True
        )


    def construct(self, x):
        batch_size = x.shape[0]
        x = ops.Transpose()(x, (0, 2, 1, 3))  # B, N, Tin, Cin
        x = ops.reshape(x,(batch_size, self.num_of_vertices, -1))
        out1 = self.fc1(x)
        relu = nn.ReLU()
        out1 = relu(out1)  # (B, N, Tin * Cin) -> (B, N, hidden)
        out2 = self.fc2(out1)  # (B, N, hidden) -> (B, N, horizon)

        return ops.Transpose()(out2, (0, 2, 1))  # B, horizon, N

class FOGS_model(AbstractModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)


        self._scaler = data_feature.get('scaler')

        self.adj_mx = Tensor(data_feature.get('adj_mx'), dtype=mstype.float32)
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.hidden_dims = config.get('hidden_dims', [[64, 64, 64], [64, 64, 64], [64, 64, 64]])
        self.first_layer_embedding_size = config.get('first_layer_embedding_size', 64)
        self.out_layer_dim = config.get('out_layer_dim', 128)
        self.activation = config.get('activation', 'GLU')
        self.use_mask = config.get('use_mask', True)
        self.temporal_emb = config.get('temporal_emb', True)
        self.spatial_emb = config.get('spatial_emb', True)
        self.strides = config.get('strides', 3)
        self.use_trend = config.get('use_trend', True)
        self.adj = self.adj_mx
        self.num_of_vertices = self.num_nodes
        self.horizon = self.output_window

        self.default_loss_function = SmoothL1Loss()
        history = self.input_window
        out_layer_dim = self.out_layer_dim
        first_layer_embedding_size = self.first_layer_embedding_size
        in_dim = self.feature_dim
        self.First_FC = nn.Dense(in_dim, first_layer_embedding_size)

        self.STSGCLS = nn.CellList()
        self.STSGCLS.append(
            STSGCL(self.adj, history, self.num_of_vertices, first_layer_embedding_size, self.hidden_dims[0],
                   self.strides, self.activation, self.temporal_emb, self.spatial_emb))

        in_dim = self.hidden_dims[0][-1]
        history -= (self.strides - 1)

        for idx, hidden_list in enumerate(self.hidden_dims):
            if idx == 0:
                continue
            self.STSGCLS.append(
                STSGCL(self.adj, history, self.num_of_vertices, in_dim, hidden_list, self.strides, self.activation,
                       self.temporal_emb, self.spatial_emb))

            history -= (self.strides - 1)
            in_dim = hidden_list[-1]

        self.predictLayer = nn.CellList()
        for t in range(self.horizon):
            self.predictLayer.append(output_layer(self.num_of_vertices, history, in_dim, out_layer_dim, 1))

        if self.use_mask:
            mask = Tensor((self.adj != 0).astype(np.float32), dtype=mstype.float32)
            self.mask = mask
        else:
            self.mask = None

    def construct(self, x):

        first_fc_output = self.First_FC(x)
        relu_op = nn.ReLU()

        x = relu_op(first_fc_output)

        for model in self.STSGCLS:
            x = model(x, self.mask)

        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](x)  # (B, 1, N)
            need_concat.append(out_step)
        out = ops.Concat(axis=1)(need_concat)


        return out

class FOGS(nn.Cell):
    def __init__(self, config, data_feature):
        super(FOGS, self).__init__()
        # data feature
        self._scaler = data_feature['scaler']
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self._logger = getLogger()

        # model config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.hidden_dims = config.get('hidden_dims', [[64, 64, 64], [64, 64, 64], [64, 64, 64]])
        self.first_layer_embedding_size = config.get('first_layer_embedding_size', 64)
        self.out_layer_dim = config.get('out_layer_dim', 128)
        self.activation = config.get('activation', 'GLU')
        self.use_mask = config.get('use_mask', True)
        self.temporal_emb = config.get('temporal_emb', True)
        self.spatial_emb = config.get('spatial_emb', True)
        self.strides = config.get('strides', 3)
        self.use_trend = config.get('use_trend', True)
        self.trend_embedding = config.get('trend_embedding', False)
        if self.trend_embedding:
            self.trend_bias_embeddings = nn.Embedding(288, self.num_nodes * self.output_window)
        self.adj = self.adj_mx
        self.num_of_vertices = self.num_nodes
        self.horizon = self.output_window

        self.network = FOGS_model(config, data_feature)
        self.default_loss_function = SmoothL1Loss()
        history = self.input_window
        out_layer_dim = self.out_layer_dim

        in_dim = self.feature_dim
        first_layer_embedding_size = self.first_layer_embedding_size

        # 权重和偏置初始化器
        first_fc_weight =  initializer(XavierNormal(gain=0.0003),
                    shape=[first_layer_embedding_size, in_dim], dtype=mindspore.float32)
        #XavierNormal(gain=0.0003)
        bias_init = initializer(Uniform(),
                    shape=[first_layer_embedding_size],dtype=mindspore.float32)

        # 使用初始化器创建权重和偏置的Tensor
        weight_shape = (first_layer_embedding_size, in_dim)
        bias_shape = (first_layer_embedding_size,)

        self.first_fc_weight = Tensor(
            initializer(first_fc_weight, shape=weight_shape, dtype=mindspore.float32)
        )
        self.first_fc_bias = Tensor(
            initializer(bias_init, shape=bias_shape, dtype=mindspore.float32)
        )

        self.First_FC = nn.Dense(
            in_channels=in_dim,
            out_channels=first_layer_embedding_size,
            weight_init=self.first_fc_weight,
            bias_init=self.first_fc_bias,
            has_bias=True
        )

        self.STSGCLS = nn.CellList()
        self.STSGCLS.append(
            STSGCL(
                adj=self.adj,
                history=history,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                out_dims=self.hidden_dims[0],
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )
        )

        in_dim = self.hidden_dims[0][-1]
        history -= (self.strides - 1)

        for idx, hidden_list in enumerate(self.hidden_dims):
            if idx == 0:
                continue
            self.STSGCLS.append(
                STSGCL(
                    adj=self.adj,
                    history=history,
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,
                    out_dims=hidden_list,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb
                )
            )

            history -= (self.strides - 1)
            in_dim = hidden_list[-1]

        self.predictLayer = nn.CellList()
        for t in range(self.horizon):
            self.predictLayer.append(
                output_layer(
                    num_of_vertices=self.num_of_vertices,
                    history=history,
                    in_dim=in_dim,
                    hidden_dim=out_layer_dim,
                    horizon=1
                )
            )

        if self.use_mask:
            mask = ops.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = Parameter(mask,requires_grad=True)
        else:
            self.mask = None


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

    def forward(self, x):
        x = self.First_FC(x)
        relu = nn.ReLU()
        x = relu(x)

        for model in self.STSGCLS:
            x = model(x, self.mask)

        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](x)
            need_concat.append(out_step)

        out = ops.Concat(axis=1)(need_concat)  # (B, Tout, N)
        del need_concat

        return out


    def _compute_embedding_loss(self, x, y_true, y_pred, bias, null_val=np.nan):

        x = x.squeeze()  # (B, T, N, 1) -> (B, T, N)
        x_truth = self.scaler.inverse_transform(x)

        x_truth = ops.Transpose()(x_truth, (1, 0, 2))  # (B, T, N) -> (T, B, N)
        y_true = ops.Transpose()(y_true, (1, 0, 2))
        y_pred = ops.Transpose()(y_pred, (1, 0, 2))

        labels = (1 + y_true) * x_truth[-1] - (1 + y_pred) * x_truth[-1]  # (T, B, N)
        labels = ops.Transpose()(labels, (1, 0, 2))  # (T, B, N) -> (B, T, N)

        return loss.masked_mae_m(bias, labels, null_val)

    def predict(self,batch_x, batch_y, batch_extx, batch_exty):
        y_predict = self.forward(batch_x)
        return y_predict


    def calculate_loss(self, batch_x, batch_y, batch_extx, batch_exty):
        input = batch_x[:, :, :, 0]
        real_val = batch_y[:, :, :, 0]

        realy_slot = batch_exty
        output = self.predict(batch_x, batch_y, batch_extx, batch_exty)
        if self.trend_embedding:
            trend_time_bias = self.trend_bias_embeddings(realy_slot[:, 0])  # (B, N * T)
            trend_time_bias = ops.Reshape(trend_time_bias, (-1, self.num_nodes, self.horizon))  # (B, N, T)
            return loss.masked_mae_m(output, real_val) + self._compute_embedding_loss(input, real_val, output,
                                                                                          trend_time_bias)

        else:
            if self.use_trend:
                return loss.masked_mae_m(output, real_val)
            else:
                output = self._scaler.inverse_transform(output)
                return self.default_loss_function(output, real_val)

    def construct(self, batch_x, batch_y, batch_extx, batch_exty):
        if self.mode=="train":
            return self.calculate_loss(batch_x, batch_y, batch_extx, batch_exty)
        elif self.mode=="eval":
            return self.predict(batch_x, batch_y, batch_extx, batch_exty)
        return
