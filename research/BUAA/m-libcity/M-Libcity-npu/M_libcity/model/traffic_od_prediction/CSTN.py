import mindspore
import mindspore.nn as nn
import time
import numpy as np

from model import loss
from model.abstract_traffic_state_model import AbstractTrafficStateModel


class ConvLSTMCell(nn.Cell):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2)
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              pad_mode='pad',
                              padding=self.padding,
                              has_bias=self.bias)

    def construct(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state



        combined = mindspore.ops.concat([input_tensor, h_cur], axis=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        temp=mindspore.numpy.split(combined_conv, 4, axis=1)

        cc_i, cc_f, cc_o, cc_g =temp[0], temp[1], temp[2], temp[3]
        sigmoid = mindspore.ops.Sigmoid()
        tanh = mindspore.ops.Tanh()
        i = sigmoid(cc_i)
        f = sigmoid(cc_f)
        o = sigmoid(cc_o)
        g = tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (mindspore.numpy.zeros([batch_size, self.hidden_dim, height, width]),
                mindspore.numpy.zeros([batch_size, self.hidden_dim, height, width]))


class ConvLSTM(nn.Cell):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.CellList(cell_list)

    def construct(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.transpose(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.shape

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.shape[1]
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = mindspore.numpy.stack(output_inner, axis=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class CNN(nn.Cell):
    def __init__(self, height, width, n_layers):
        super(CNN, self).__init__()
        self.height = height
        self.width = width
        self.n_layers = n_layers
        self.conv = nn.CellList()
        self.conv.append(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), pad_mode='pad', padding=(1, 1, 1, 1)))
        for i in range(1, self.n_layers):
            self.conv.append(
                nn.ReLU()
            )
            self.conv.append(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), pad_mode='pad', padding=(1, 1, 1, 1))
            )
        self.relu = nn.ReLU()
        self.embed = nn.SequentialCell(
            nn.Conv2d(in_channels=self.height * self.width * 16, out_channels=32, kernel_size=(1, 1)),
            nn.ReLU()
        )

    def construct(self, x):
        # (B, T, H, W, H, W)
        x = x.reshape(-1, 1, self.height, self.width)
        # (B * T * H * W, 1, H, W)
        _x = x
        x = self.conv[0](x)
        for i in range(1, self.n_layers):
            x += _x
            x = self.conv[2 * i - 1](x)
            _x = x
            x = self.conv[2 * i](x)
        x += _x
        x = self.relu(x)
        x = x.reshape(-1, self.height * self.width * 16, self.height, self.width)
        # (B * T, H * W * 16, H, W)
        x = self.embed(x)
        # (B * T, 32, H, W)
        return x


class MLP(nn.Cell):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_1 = nn.SequentialCell(nn.Dense(in_channels=29, out_channels=32), nn.ReLU())
        self.fc_2 = nn.SequentialCell(nn.Dense(in_channels=32, out_channels=16), nn.ReLU())
        self.fc_3 = nn.SequentialCell(nn.Dense(in_channels=16, out_channels=8), nn.ReLU())

    def construct(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x


class LSC(nn.Cell):
    def __init__(self, height, width, n_layers):
        super(LSC, self).__init__()
        self.height = height
        self.width = width
        self.n_layers = n_layers
        self.O_CNN = CNN(self.height, self.width, self.n_layers)
        self.D_CNN = CNN(self.height, self.width, self.n_layers)
        self.embeder = nn.SequentialCell(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)), nn.ReLU())

    def construct(self, x, xt):
        # (B, T, H, W, H, W)
        x = self.O_CNN(x)
        xt = self.D_CNN(xt)

        x = mindspore.ops.concat([x, xt], axis=1)
        # (B * T, 64, H, W)
        x = self.embeder(x)
        # (B * T, 32, H, W)
        return x


class TEC(nn.Cell):
    def __init__(self, c_lt):
        super(TEC, self).__init__()
        self.ConvLSTM = ConvLSTM(
            input_dim=40, hidden_dim=32, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.conv = nn.SequentialCell(
            nn.Conv2d(in_channels=32, out_channels=c_lt, kernel_size=(1, 1)),
            nn.ReLU()
        )

    def construct(self, x):
        # (B, T, 40, H, W)
        (x, _) = self.ConvLSTM(x)[1][0]
        # (B, 40, H, W)
        x = self.conv(x)
        # (B, c_lt, H, W)
        return x


class GCC(nn.Cell):
    def __init__(self):
        super(GCC, self).__init__()
        self.Softmax = nn.Softmax(axis=0)

    def construct(self, x):
        # x: (B, c_lt, H, W)
        _x = x
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        bmm = mindspore.ops.BatchMatMul()
        s = bmm(x.transpose((0, 2, 1)), x)
        s = self.Softmax(s)
        # s: (B, H * W, H * W)
        x = bmm(x, s)
        # x: (B, c_lt, H, W)
        x = x.reshape(_x.shape)
        x = mindspore.ops.concat([x, _x], axis=1)
        # x: (B, 2 * c_lt, H, W)
        return x


class CSTN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(CSTN,self).__init__(config, data_feature)

        self.batch_size = config.get('batch_size')
        self.output_dim = config.get('output_dim')

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 12)
        self._scaler = self.data_feature.get('scaler')
        self.height = data_feature.get('len_row', 15)
        self.width = data_feature.get('len_column', 5)

        self.n_layers = config.get('n_layers', 3)
        self.c_lt = config.get('c_lt', 75)

        self.LSC = LSC(self.height, self.width, self.n_layers)
        self.MLP = MLP()
        self.TEC = TEC(self.c_lt)
        self.GCC = GCC()
        self.OUT = nn.SequentialCell(
            nn.Conv2d(
                in_channels=2 * self.c_lt,
                out_channels=self.height * self.width,
                kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Tanh()
        )
        self.mode = "train"

        self.loss_fn=mindspore.nn.MSELoss()

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"

    def forward(self, batch):
        x = batch[0][..., 0]
        # x : (B, T, H, W, H, W)
        s=time.time()
        xt = x.transpose((0, 1, 4, 5, 2, 3))
        x = self.LSC(x,xt)
        batch_size = np.prod(x.shape) / (self.input_window * 32 * self.height * self.width)
        x = x.reshape((int(batch_size), self.input_window, 32, self.height, self.width))

        w = batch[1]
        # w : (B, T, F)
        w = self.MLP(w)
        # w : (B, T, 8)

        w = w.repeat(self.height * self.width,axis=1)
        batch_size = np.prod(w.shape) / (self.input_window * 8 * self.height * self.width)
        w = w.reshape((int(batch_size), self.input_window, 8, self.height, self.width))
        # w : (B, T, 8, H, W)
        x = mindspore.ops.concat([x, w], axis=2)
        x = self.TEC(x)
        x = self.GCC(x)

        # x : (B, 2 * c_lt, H, W)
        x = self.OUT(x)
        # x : (B, H * W, H, W)
        batch_size = np.prod(x.shape) / (self.height ** 2 * self.width ** 2)
        x = x.reshape((int(batch_size), 1, self.height, self.width, self.height, self.width, 1))
        return x

    def set_loss(self, loss_fn):
        pass

    def calculate_loss(self, x, w, y_true):
        y_predicted = self.predict(x, w, y_true)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        res = self.loss_fn(y_predicted, y_true)
        return res

    def predict(self, x_, w, y):
        y_preds = []
        for i in range(self.output_window):
            batch_tmp = [ x_, w[:, i:(i + self.input_window), ...]]
            y_ = self.forward(batch_tmp)  # (batch_size, 1, len_row, len_column, output_dim)
            y_preds.append(y_)
            if y_.shape[-1] < x_.shape[-1]:  # output_dim < feature_dim
                y_ = mindspore.ops.concat([y_, y[:, i:i + 1, :, :, self.output_dim:]], axis=-1)
            x_ = mindspore.ops.concat([x_[:, 1:, :, :, :], y_], axis=1)
        y_preds = mindspore.ops.concat(y_preds, axis=1)  # (batch_size, output_length, len_row, len_column, output_dim)
        return y_preds

    def construct(self, x, w, y):
        x=x.astype(mindspore.float32)
        w = w.astype(mindspore.float32)
        y = y.astype(mindspore.float32)
        if self.mode == "train":
            return self.calculate_loss(x, w, y)
        elif self.mode == "eval":
            y_preds = self.predict(x, w, y)
            y_true = self._scaler.inverse_transform(y[..., :self.output_dim])
            y_preds = self._scaler.inverse_transform(y_preds[..., :self.output_dim])
            return y_preds, y_true

    
    def evaluate(self, x, w, y):
        x=x.astype(mindspore.float32)
        w = w.astype(mindspore.float32)
        y = y.astype(mindspore.float32)

        y_preds = self.predict(x, w, y)
        y_true = self._scaler.inverse_transform(y[..., :self.output_dim])
        y_preds = self._scaler.inverse_transform(y_preds[..., :self.output_dim])
        return y_preds, y_true