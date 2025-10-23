from time import time
from typing import Union
import time
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

dense_init = "normal"

class SMAPELoss(nn.Cell):

    def divide_no_nan(self, a, b):
        div = a / b
        div[div != div] = 0.0
        div[div == float('inf')] = 0.0 
        return div
    
    def construct(self, y, y_hat):
        delta_y = ops.abs(y - y_hat)
        scale = ops.abs(y) + ops.abs(y_hat)
        smape = self.divide_no_nan(delta_y, scale)
        smape = 200 * ops.mean(smape)
        return smape

class NBeatsNet(nn.Cell):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(
            self,
            stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
            nb_blocks_per_stack=3,
            forecast_length=5,
            backcast_length=10,
            thetas_dim=(4, 8),
            share_weights_in_stack=False,
            hidden_layer_units=256,
            nb_harmonics=None
    ):   
        super(NBeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dim
        self.para_prefix = 0
        self.parameters = []
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = ms.ParameterTuple(self.parameters)
        self._loss = None
        self._opt = None
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(
                    self.hidden_layer_units, self.thetas_dim[stack_id],
                    self.backcast_length, self.forecast_length,
                    self.nb_harmonics
                )
                block.update_parameters_name(prefix=f'ms{self.para_prefix}_', recurse=True)
                self.para_prefix += 1
                self.parameters.extend(block.trainable_params())
            blocks.append(block)
        return blocks
    
    def disable_intermediate_outputs(self):
        self._gen_intermediate_outputs = False

    def enable_intermediate_outputs(self):
        self._gen_intermediate_outputs = True

    def save(self, filename: str):
        ms.save_checkpoint(self, filename)

    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock
        
    def compile(self, loss: str, optimizer: Union[str, nn.Optimizer]):
        if loss == 'mae':
            loss_ = nn.L1Loss()
        elif loss == 'mse':
            loss_ = nn.MSELoss()
        elif loss == 'cross_entropy':
            loss_ = nn.CrossEntropyLoss()
        elif loss == 'binary_crossentropy':
            loss_ = ms.ops.BinaryCrossEntropy()
        elif loss == 'smape':
            loss_ = SMAPELoss()
        else:
            raise ValueError(f'Unknown loss name: {loss}.')
        # noinspection PyArgumentList
        if isinstance(optimizer, str):
            if optimizer == 'adam':
                opt_ = nn.Adam(params=self.parameters, learning_rate=1e-4)
            elif optimizer == 'sgd':
                opt_ = nn.SGD(params=self.parameters, learning_rate=1e-4)
            elif optimizer == 'rmsprop':
                opt_ = nn.RMSProp(params=self.parameters, learning_rate=1e-4)
            else:
                raise ValueError(f'Unknown opt name: {optimizer}.')
        else:
            opt_ = opt_(params=self.parameters, learning_rate=1e-4)
        self._opt = opt_
        self._loss = loss_
    
    def fit(self, train_data, test_data, epochs=200, sv_name="tmp", reducer_flag=False):
        if reducer_flag:
            mean = ms.context.get_auto_parallel_context("gradients_mean")
            degree = ms.context.get_auto_parallel_context("device_num")
            grad_reducer = ms.nn.DistributedGradReducer(self._opt.parameters, mean, degree)
        
        for epoch in range(epochs):
            iter_count = 0
            train_loss = []
            test_loss = []
            self.set_train()
            epoch_time = time.time()

            def forward_fn(seq_x, true):
                pred = self(seq_x)
                loss = self._loss(pred, true)
                return loss, true

            for i, (seq_x, seq_y) in enumerate(train_data.create_tuple_iterator()):
                iter_count += 1
                grad_fn = ops.value_and_grad(forward_fn, None, self._opt.parameters, has_aux=True)
                (loss, _), grads = grad_fn(seq_x, seq_y)
                if reducer_flag:
                    grads = grad_reducer(grads)
                loss = ops.depend(loss, self._opt(grads))
                train_loss.append(loss.asnumpy())
            train_loss = np.average(train_loss)    

            if epoch % 10 == 0 or epoch == epochs - 1:
                self.set_train(False)
                for i, (seq_x, seq_y) in enumerate(test_data.create_tuple_iterator()):
                    iter_count += 1
                    loss, _ = forward_fn(seq_x, seq_y)
                    test_loss.append(loss.asnumpy())
                test_loss = np.average(test_loss)
                print("Epoch: {0} cost time: {1} Train Loss: {2:.4f} Test Loss: {3:.4f}".format(
                        epoch+1, time.time() - epoch_time, train_loss, test_loss))
            else:
                print("Epoch: {0} cost time: {1} Train Loss: {2:.4f}".format(
                        epoch+1, time.time() - epoch_time, train_loss))
        ms.save_checkpoint(self, f"./checkpoints/train_ckpt/{sv_name}.ckpt")

    def test(self, test_data, ckpt_path=None):
        test_loss = []
        def forward_fn(seq_x, true):
            pred = self(seq_x)
            loss = self._loss(pred, true)
            return loss, true
        self.set_train(False)
        for i, (seq_x, seq_y) in enumerate(test_data.create_tuple_iterator()):
            loss, _ = forward_fn(seq_x, seq_y)
            test_loss.append(loss.asnumpy())
        test_loss = np.average(test_loss)
        print("Test Loss: {:.4f}".format(test_loss))
        
    @staticmethod
    def name():
        return 'NBeatsMindspore'

    def get_generic_and_interpretable_outputs(self):
        g_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' in a['layer'].lower()])
        i_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' not in a['layer'].lower()])
        outputs = {o['layer']: o['value'][0] for o in self._intermediary_outputs}
        return g_pred, i_pred, outputs

    def construct(self, backcast):
        self._intermediary_outputs = []
        backcast = squeeze_last_dim(backcast)
        forecast = ms.ops.Zeros()((backcast.shape[0], self.forecast_length), ms.float32)  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                forecast = forecast + f
                block_type = self.stacks[stack_id][block_id].__class__.__name__
                layer_name = f'stack_{stack_id}-{block_type}_{block_id}'
                if self._gen_intermediate_outputs:
                    self._intermediary_outputs.append({'value': f.detach().numpy(), 'layer': layer_name})
        return forecast


def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor[..., 0]
    return tensor

def seasonality_model(thetas, t):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = ms.Tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float()  # H/2-1
    s2 = ms.Tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
    S = ms.ops.cat([s1, s2])
    return thetas.mm(S)

def trend_model(thetas, t):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = ms.Tensor(np.array([t ** i for i in range(p)])).float()
    return thetas.mm(T)

def linear_space(backcast_length, forecast_length, is_forecast=True):
    horizon = forecast_length if is_forecast else backcast_length
    return np.arange(0, horizon) / horizon

class Block(nn.Cell):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, share_thetas=False,
                 nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Dense(backcast_length, units, weight_init=dense_init)
        self.fc2 = nn.Dense(units, units, weight_init=dense_init)
        self.fc3 = nn.Dense(units, units, weight_init=dense_init)
        self.fc4 = nn.Dense(units, units, weight_init=dense_init)
        self.backcast_linspace = linear_space(backcast_length, forecast_length, is_forecast=False)
        self.forecast_linspace = linear_space(backcast_length, forecast_length, is_forecast=True)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Dense(units, thetas_dim, has_bias=False, weight_init=dense_init)
        else:
            self.theta_b_fc = nn.Dense(units, thetas_dim, has_bias=False, weight_init=dense_init)
            self.theta_f_fc = nn.Dense(units, thetas_dim, has_bias=False, weight_init=dense_init)
        self.relu = ms.ops.ReLU()

    def construct(self, x):
        x = squeeze_last_dim(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, backcast_length,
                                                   forecast_length, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, forecast_length, backcast_length,
                                                   forecast_length, share_thetas=True)

    def construct(self, x):
        x = super(SeasonalityBlock, self).construct(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(TrendBlock, self).__init__(units, thetas_dim, backcast_length,
                                         forecast_length, share_thetas=True)

    def construct(self, x):
        x = super(TrendBlock, self).construct(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(GenericBlock, self).__init__(units, thetas_dim, backcast_length, forecast_length)

        self.backcast_fc = nn.Dense(thetas_dim, backcast_length, weight_init=dense_init)
        self.forecast_fc = nn.Dense(thetas_dim, forecast_length, weight_init=dense_init)

    def construct(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).construct(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast
