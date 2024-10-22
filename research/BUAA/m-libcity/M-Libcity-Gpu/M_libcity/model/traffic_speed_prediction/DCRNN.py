import numpy as np
from mindspore import Tensor

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer
import model.loss
import model.utils as utils

# import dcrnn_util as utils

count_weight = 0
count_bias = 0


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Cell, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        super(EncoderModel, self).__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.dcgru_layers = nn.CellList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def construct(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.shape
        if hidden_state is None:
            hidden_state = ops.Zeros()((self.num_rnn_layers, batch_size, self.hidden_state_size), mindspore.float32)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        out_stack = ops.Stack()(hidden_states)
        return output, out_stack  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Cell, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        super(DecoderModel, self).__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Dense(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.CellList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def construct(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))

        output = projected.view(-1, self.num_nodes * self.output_dim)

        out_stack = ops.Stack()(hidden_states)
        return output, out_stack


class DCRNNModel(nn.Cell, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        super(DCRNNModel, self).__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self.batches_seen = 0

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.shape[1]
        go_symbol = ops.Zeros()((batch_size, self.num_nodes * self.decoder_model.output_dim), mindspore.float32)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []
        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = ops.Stack()(outputs)
        return outputs

    def construct(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        if batches_seen is not None:
            self.batches_seen += 32
        encoder_hidden_state = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=self.batches_seen)

        return outputs


class LayerParams:
    def __init__(self, rnn_network: nn.Cell, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            global count_weight
            count_weight += 1
            nn_param = mindspore.Parameter(initializer('xavier_uniform', shape, mindspore.float32))
            self._params_dict[shape] = nn_param
            self._rnn_network.insert_param_to_cell('{}_weight_{}_{}'.format(self._type, str(shape), str(count_weight)),
                                                   nn_param)
            print('new weight:', shape)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            global count_bias
            count_bias += 1
            biases = mindspore.Parameter(initializer(bias_start, length, mindspore.float32))
            self._biases_dict[length] = biases
            self._rnn_network.insert_param_to_cell('{}_biases_{}_{}'.format(self._type, str(length), str(count_bias)),
                                                   biases)
            print('new bias:', length)
        return self._biases_dict[length]


class DCGRUCell(nn.Cell):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = ops.Tanh() if nonlinearity == 'tanh' else ops.ReLU()
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        supports = []
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(mindspore.Tensor(support.toarray(), mindspore.float32))

        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')

    def construct(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = ops.Sigmoid()(fn(inputs, hx, output_size, bias_start=1.0))
        value = ops.Reshape()(value, (-1, self._num_nodes, output_size))
        r, u = value[:, :, :int(value.shape[-1] / 2)], value[:, :, int(value.shape[-1] / 2):]
        r = ops.Reshape()(r, (-1, self._num_nodes * self._num_units))
        u = ops.Reshape()(u, (-1, self._num_nodes * self._num_units))
        c = self._gconv(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)
        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.expand_dims(axis=0)
        return ops.Concat(axis=0)([x, x_])

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = ops.Reshape()(inputs, (batch_size * self._num_nodes, -1))
        state = ops.Reshape()(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = ops.Concat(axis=-1)([inputs, state])
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = ops.Sigmoid()(ops.MatMul()(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = ops.Reshape()(inputs, (batch_size, self._num_nodes, -1))
        state = ops.Reshape()(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = ops.Concat(axis=2)([inputs, state])
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state
        x0 = x.transpose(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = ops.Reshape()(x0, (self._num_nodes, input_size * batch_size))
        x = ops.ExpandDims()(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = ops.MatMul()(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * ops.MatMul()(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
        x = ops.Reshape()(x, (num_matrices, self._num_nodes, input_size, batch_size))
        x = x.transpose(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = ops.Reshape()(x, (batch_size * self._num_nodes, input_size * num_matrices))

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = ops.MatMul()(x, weights)
        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        return ops.Reshape()(x, (batch_size, self._num_nodes * output_size))

class DCRNN(nn.Cell):
    def __init__(self, config, data_feature):
        super(DCRNN, self).__init__()
        self.loss = model.loss.masked_mae_m
        self.zscore = data_feature['scaler']

        kwargs = {'cl_decay_steps': config['cl_decay_steps'], 'filter_type': config['filter_type'],
                  'horizon': config['input_window'],
                  'input_dim': data_feature['feature_dim'],
                  'l1_decay': 0,
                  'max_diffusion_step': config['max_diffusion_step'], 'num_nodes': data_feature['num_nodes'],
                  'num_rnn_layers': config['num_rnn_layers'], 'output_dim': 1, 'rnn_units': config['rnn_units'],
                  'seq_len': config['input_window'],
                  'use_curriculum_learning': config['use_curriculum_learning']}
        print(kwargs)
        print((kwargs['seq_len'], kwargs['rnn_units'], kwargs['num_nodes'] * 2))
        adj_mx = data_feature['adj_mx']
        self.input_window=config['input_window']
        self.network = DCRNNModel(adj_mx, **kwargs)
        self.network.set_train(False)
        self.network(
            mindspore.ops.Zeros()((kwargs['seq_len'], kwargs['rnn_units'], kwargs['num_nodes'] * 2), mindspore.float32))
        self.network.set_train(True)
        self.reshape = ops.Reshape()
        self.output_window = config.get('output_window', 12)
        
        self.mode = "train"

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        
        x = x.transpose(1, 0, 2, 3)
        i,b,_,_=x.shape
        x = x.reshape([i,b,-1])
        y = y.transpose(1, 0, 2, 3)
        y = y[...,0]
        return x, y

    def set_loss(self, loss_fn):
        pass

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"
        
    def calculate_loss(self, x, label):
        y = label
        x, y = self._get_x_y(x, y)
        x = self.network(x, y, 1)
        x = self.zscore.inverse_transform(x)
        y = self.zscore.inverse_transform(y)
        loss = self.loss(x, y, 0.0)
        return loss

    def predict(self, x, label):
        y = label
        x, y = self._get_x_y(x, y)
        x = self.network(x, y, 1)
        x = self.zscore.inverse_transform(x)
        y = self.zscore.inverse_transform(y)
        x = x.transpose(1, 0, 2)
        y = y.transpose(1, 0, 2)
        return x, y
 
    def construct(self, x, label):
        x = x.astype(dtype=mindspore.float32)
        label = label.astype(dtype=mindspore.float32)
        if self.mode == "train":
            return self.calculate_loss(x, label)
        elif self.mode == "eval":
            return self.predict(x, label)
        