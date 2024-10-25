import mindspore
import mindspore.nn as nn
from collections import OrderedDict
import mindspore.ops as ops
import mindspore.numpy as mnp
from model.abstract_model import AbstractModel
import model.loss

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, pad_mode='pad', padding=1, has_bias=True)


class BnReluConv(nn.Cell):
    """
    # 1.batchNorm2d(selected, default:False)
    # 2.relu
    # 3.conv1
    """

    def __init__(self, num_channels, bn=False):
        super(BnReluConv, self).__init__()
        self.has_bn = bn
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.conv1 = conv3x3(num_channels, num_channels)

    def construct(self, x):
        if self.has_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x


class ResidualUnit(nn.Cell):
    def __init__(self, num_channels, bn=False):
        super(ResidualUnit, self).__init__()
        self.bn_relu_conv1 = BnReluConv(num_channels, bn)
        self.bn_relu_conv2 = BnReluConv(num_channels, bn)

    def construct(self, x):
        residual = x
        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)
        out += residual
        return out


class ResUnits(nn.Cell):
    def __init__(self, residual_unit, num_channels, repetitions=1, bn=False):
        super(ResUnits, self).__init__()
        self.residual_units = self.make_residual_units(residual_unit, num_channels, repetitions, bn)

    def make_residual_units(self, residual_unit, num_channels, repetitions, bn):
        layers = []
        for i in range(repetitions):
            layers.append(residual_unit(num_channels, bn))
        return nn.SequentialCell(layers)

    def construct(self, x):
        out = self.residual_units(x)
        return out


class TrainableEltwiseLayer(nn.Cell):
    # Matrix-based fusion
    def __init__(self, n, h, w):
        super(TrainableEltwiseLayer, self).__init__()
        self.weights = mindspore.Parameter(mindspore.numpy.randn(1, n, h, w),
                                           requires_grad=True)  # define the trainable parameter

    def construct(self, x):
        # assuming x is of size b-1-h-w
        x = x * self.weights  # element-wise multiplication
        return x

class STResNet_model(AbstractModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.nb_residual_unit = config.get('nb_residual_unit', 12)
        self.bn = config.get('batch_norm', False)
        #device
        self._scaler = data_feature['scaler']
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 2)#inflow and outflow
        self.ext_dim = data_feature.get('ext_dim', 0)
        self.output_dim = data_feature.get('output_dim', 2)
        self.len_row = data_feature.get('len_row', 32)
        self.len_column = data_feature.get('len_column', 32)
        self.len_closeness = data_feature.get('len_closeness', 4)
        self.len_period = data_feature.get('len_period', 2)
        self.len_trend = data_feature.get('len_trend', 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        if self.len_closeness > 0:
            # 数据在第一个维度上经过拼接
            self.c_way = self.make_one_way(in_channels=self.len_closeness * self.feature_dim)

        if self.len_period > 0:
            self.p_way = self.make_one_way(in_channels=self.len_period * self.feature_dim)

        if self.len_trend > 0:
            self.t_way = self.make_one_way(in_channels=self.len_trend * self.feature_dim)

        if self.ext_dim > 0:
            self.external_ops = nn.SequentialCell(OrderedDict([
                ('embd', nn.Dense(self.ext_dim, 10, has_bias=True)),
                ('relu1', nn.ReLU()),
                ('fc', nn.Dense(10, self.output_dim * self.len_row * self.len_column, has_bias=True)),
                ('relu2', nn.ReLU())
            ]))
            
    # make a way for closeness/ period/ trend
    def make_one_way(self, in_channels):
        return nn.SequentialCell(OrderedDict([
            ('conv1', conv3x3(in_channels=in_channels, out_channels=64)),
            # 可能需要relu
            ('ResUnits', ResUnits(ResidualUnit, num_channels=64, repetitions=self.nb_residual_unit, bn=self.bn)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(in_channels=64, out_channels=2)),
            ('FusionLayer', TrainableEltwiseLayer(n=self.output_dim, h=self.len_row, w=self.len_column))

        ]))
            
    def construct(self, batch_x, batch_ext):
        inputs = batch_x  # (batch_size, T_c+T_p+T_t, len_row, len_column, feature_dim)
        input_ext = batch_ext  # (batch_size, ext_dim)
        #input_ext = None
        batch_size, len_time, len_row, len_column, input_dim = inputs.shape

        main_output = 0
        if self.len_closeness > 0:
            begin_index = 0
            end_index = begin_index + self.len_closeness
            input_c = inputs[:, begin_index:end_index, :, :, :]
            #实现维度的拼接，len_closeness 为 lc
            input_c = input_c.view(-1, self.len_closeness * self.feature_dim, self.len_row, self.len_column)
            out_c = self.c_way(input_c)
            main_output += out_c
        if self.len_period > 0:
            begin_index = self.len_closeness
            end_index = begin_index + self.len_period
            input_p = inputs[:, begin_index:end_index, :, :, :]
            input_p = input_p.view(-1, self.len_period * self.feature_dim, self.len_row, self.len_column)
            out_p = self.p_way(input_p)
            main_output += out_p
        if self.len_trend > 0:
            begin_index = self.len_closeness + self.len_period
            end_index = begin_index + self.len_trend
            input_t = inputs[:, begin_index:end_index, :, :, :]
            input_t = input_t.view(-1, self.len_trend*self.feature_dim, self.len_row, self.len_column)
            out_t = self.t_way(input_t)
            main_output += out_t
        if self.ext_dim > 0:
            external_output = self.external_ops(input_ext)
            external_output = self.relu(external_output)
            external_output = external_output.view(-1, self.feature_dim, self.len_row, self.len_column)
            main_output += external_output
        main_output = self.tanh(main_output)
        main_output = main_output.view(batch_size, -1, len_row, len_column, self.output_dim)
        return main_output

class STResNet(nn.Cell):
    def __init__(self, config, data_feature):
        super(STResNet, self).__init__() 
        self.loss = model.loss.masked_mse_m
        self.network = STResNet_model(config, data_feature)
        self.reshape = ops.Reshape()
        self.zscore = data_feature['scaler']
        
    def train(self):
        self.mode="train"

    def eval(self):
        self.mode="eval"

    def construct(self, batch_x, batch_y, batch_extx, batch_exty):
        if self.mode=="train":
            return self.calculate_loss(batch_x, batch_y, batch_extx, batch_exty)
        elif self.mode=="eval":
            return self.predict(batch_x, batch_exty, batch_y)
        return 
    
    def calculate_loss(self, batch_x, batch_y, batch_extx, batch_exty):
        x = batch_x
        x_ext = batch_extx
        y = batch_y
        y_ext = batch_exty
        y_pred = self.network(x, y_ext)
        y_pred = self.zscore.inverse_transform(y_pred)
        y = self.zscore.inverse_transform(y)
        loss = self.loss(y_pred, y)
        return loss

    def predict(self, x, batch_exty, batch_y):
        y_predict = self.network(x, batch_exty)
        y_predict = self.zscore.inverse_transform(y_predict)
        label = self.zscore.inverse_transform(batch_y)
        return y_predict, label