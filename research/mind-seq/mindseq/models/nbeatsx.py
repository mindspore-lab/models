import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from typing import Tuple
import time
import random
from functools import partial
from ..utils.nbeatsx_utils import MAELoss
from ..utils.nbeatsx_utils import mae
from mindspore.amp import StaticLossScaler

class Chomp1d(nn.Cell):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def construct(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Cell):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation, pad_mode='pad',
                               weight_init='normal', has_bias=True)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation, pad_mode='pad',
                               weight_init='normal', has_bias=True)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

        self.net = nn.SequentialCell(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, weight_init='normal', pad_mode='pad', has_bias=True) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Cell):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.network(x)

def filter_input_vars(insample_y, insample_x_t, outsample_x_t, t_cols, include_var_dict):
    # This function is specific for the EPF task
    outsample_y = ms.ops.zeros((insample_y.shape[0], 1, outsample_x_t.shape[2]), dtype=ms.float32)
    insample_y_aux = ms.ops.unsqueeze(insample_y,dim=1)
    insample_x_t_aux = ms.ops.concat([insample_y_aux, insample_x_t], axis=1)
    outsample_x_t_aux = ms.ops.concat([outsample_y, outsample_x_t], axis=1)
    x_t = ms.ops.concat([insample_x_t_aux, outsample_x_t_aux], axis=-1)
    batch_size, n_channels, input_size = x_t.shape

    assert input_size==168+24, f'input_size {input_size} not 168+24'

    x_t = x_t.reshape(batch_size, n_channels, 8, 24)

    input_vars = []
    for var in include_var_dict.keys():
        if len(include_var_dict[var])>0:
            t_col_idx    = t_cols.index(var)
            t_col_filter = include_var_dict[var].copy()
            if var != 'week_day':
                input_vars  += [x_t[:, t_col_idx, t_col_filter, :]]
            else:
                assert t_col_filter == [-1], f'Day of week must be of outsample not {t_col_filter}'
                day_var = x_t[:, t_col_idx, t_col_filter, [0]]
                day_var = day_var.view(batch_size, -1)
    x_t_filter = ms.ops.concat(input_vars, axis=1)
    x_t_filter = x_t_filter.view(batch_size,-1)

    if len(include_var_dict['week_day'])>0:
        d_a, d_b = day_var.shape
        x_a, x_b = x_t_filter.shape
        res = ms.ops.zeros((d_a, d_b + x_b), ms.float32)
        res[:,-1:] = day_var
        res[:,:-1] = x_t_filter
        # x_t_filter = ms.ops.cat((x_t_filter, day_var), axis=1)
        x_t_filter = res
    return x_t_filter

class _StaticFeaturesEncoder(nn.Cell):
    def __init__(self, in_features, out_features):
        super(_StaticFeaturesEncoder, self).__init__()
        layers = [nn.Dropout(p=0.5),
                  nn.Dense(in_channels=in_features, out_channels=out_features),
                  nn.ReLU()]
        self.encoder = nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.encoder(x)
        return x

class NBeatsBlock(nn.Cell):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self, x_t_n_inputs: int, x_s_n_inputs: int, x_s_n_hidden: int, theta_n_dim: int, basis: nn.Cell,
                 n_layers: int, theta_n_hidden: list, include_var_dict, t_cols, batch_normalization: bool, dropout_prob: float, activation: str):
        """
        """
        super().__init__()

        if x_s_n_inputs == 0:
            x_s_n_hidden = 0
        theta_n_hidden = [x_t_n_inputs + x_s_n_hidden] + theta_n_hidden

        self.x_s_n_inputs = x_s_n_inputs
        self.x_s_n_hidden = x_s_n_hidden
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        self.activations = {'relu': nn.ReLU(),
                            # 'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SeLU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}

        hidden_layers = []
        for i in range(n_layers):

            # Batch norm after activation
            hidden_layers.append(nn.Dense(in_channels=theta_n_hidden[i], out_channels=theta_n_hidden[i+1]))
            hidden_layers.append(self.activations[activation])

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=theta_n_hidden[i+1]))

            if self.dropout_prob>0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Dense(in_channels=theta_n_hidden[-1], out_channels=theta_n_dim)]
        layers = hidden_layers + output_layer

        # x_s_n_inputs is computed with data, x_s_n_hidden is provided by user, if 0 no statics are used
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(in_features=x_s_n_inputs, out_features=x_s_n_hidden)
        self.layers = nn.SequentialCell(*layers)
        self.basis = basis

    def construct(self, insample_y: ms.Tensor, insample_x_t: ms.Tensor,
                outsample_x_t: ms.Tensor, x_s: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:

        if self.include_var_dict is not None:
            insample_y = filter_input_vars(insample_y=insample_y, insample_x_t=insample_x_t, outsample_x_t=outsample_x_t,
                                           t_cols=self.t_cols, include_var_dict=self.include_var_dict)

        # Static exogenous
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            x_s = self.static_encoder(x_s)
            insample_y = ms.ops.concat((insample_y, x_s), axis=1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)

        return backcast, forecast

class NBeats(nn.Cell):
    """
    N-Beats Model.
    """
    def __init__(self, blocks: ms.nn.SequentialCell):
        super().__init__()
        self.blocks = blocks

    def construct(self, insample_y: ms.Tensor, insample_x_t: ms.Tensor, insample_mask: ms.Tensor,
                outsample_x_t: ms.Tensor, x_s: ms.Tensor, return_decomposition = False):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:] # Level with Naive1
        block_forecasts = []
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
                                             outsample_x_t=outsample_x_t, x_s=x_s)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        # (n_batch, n_blocks, n_time)
        block_forecasts = ms.ops.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1,0,2)

        if return_decomposition:
            return forecast, block_forecasts
        else:
            return forecast

    def decomposed_prediction(self, insample_y: ms.Tensor, insample_x_t: ms.Tensor, insample_mask: ms.Tensor,
                              outsample_x_t: ms.Tensor):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:] # Level with Naive1
        forecast_components = []
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals, insample_x_t, outsample_x_t)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            forecast_components.append(block_forecast)
        return forecast, forecast_components

class IdentityBasis(nn.Cell):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def construct(self, theta: ms.Tensor, insample_x_t: ms.Tensor, outsample_x_t: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, -self.forecast_size:]
        return backcast, forecast

class TrendBasis(nn.Cell):
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        polynomial_size = degree_of_polynomial + 1
        self.backcast_basis = nn.Parameter(
            ms.Tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                    for i in range(polynomial_size)]), dtype=ms.float32), requires_grad=False)
        self.forecast_basis = nn.Parameter(
            ms.Tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                    for i in range(polynomial_size)]), dtype=ms.float32), requires_grad=False)

    def construct(self, theta: ms.Tensor, insample_x_t: ms.Tensor, outsample_x_t: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = ms.ops.matmul(theta[:, cut_point:], self.backcast_basis)
        forecast = ms.ops.matmul(theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast

class SeasonalityBasis(nn.Cell):
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        frequency = np.append(np.zeros(1, dtype=np.float32),
                                        np.arange(harmonics, harmonics / 2 * forecast_size,
                                                    dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * frequency

        backcast_cos_template = ms.Tensor(np.transpose(np.cos(backcast_grid)), dtype=ms.float32)
        backcast_sin_template = ms.Tensor(np.transpose(np.sin(backcast_grid)), dtype=ms.float32)
        backcast_template = ms.ops.concat([backcast_cos_template, backcast_sin_template], axis=0)

        forecast_cos_template = ms.Tensor(np.transpose(np.cos(forecast_grid)), dtype=ms.float32)
        forecast_sin_template = ms.Tensor(np.transpose(np.sin(forecast_grid)), dtype=ms.float32)
        forecast_template = ms.ops.concat([forecast_cos_template, forecast_sin_template], axis=0)

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def construct(self, theta: ms.Tensor, insample_x_t: ms.Tensor, outsample_x_t: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = ms.ops.matmul(theta[:, cut_point:], self.backcast_basis)
        forecast = ms.ops.matmul(theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast

class ExogenousBasisInterpretable(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, theta: ms.Tensor, insample_x_t: ms.Tensor, outsample_x_t: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        backcast_basis = insample_x_t
        forecast_basis = outsample_x_t

        cut_point = forecast_basis.shape[1]
        a = ms.ops.expand_dims(theta[:, cut_point:], 1)
        backcast = ms.ops.squeeze(ms.ops.matmul(a, backcast_basis))
        b = ms.ops.expand_dims(theta[:, :cut_point], 1)
        forecast = ms.ops.squeeze(ms.ops.matmul(b, forecast_basis))
        return backcast, forecast

class ExogenousBasisWavenet(nn.Cell):
    def __init__(self, out_features, in_features, num_levels=4, kernel_size=3, dropout_prob=0):
        super().__init__()
        # Shape of (1, in_features, 1) to broadcast over b and t
        self.weight = nn.Parameter(ms.Tensor(1, in_features, 1), requires_grad=True)

        padding = (kernel_size - 1) * (2**0)
        input_layer = [nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                 kernel_size=kernel_size, padding=padding, dilation=2**0, pad_mode='pad', has_bias=True),
                                 Chomp1d(padding),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_prob)]
        conv_layers = []
        for i in range(1, num_levels):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            conv_layers.append(nn.Conv1d(in_channels=out_features, out_channels=out_features,
                                         padding=padding, kernel_size=3, dilation=dilation, pad_mode='pad', has_bias=True))
            conv_layers.append(Chomp1d(padding))
            conv_layers.append(nn.ReLU())
        conv_layers = input_layer + conv_layers

        self.wavenet = nn.SequentialCell(*conv_layers)

    def transform(self, insample_x_t, outsample_x_t):
        input_size = insample_x_t.shape[2]

        x_t = ms.ops.concat([insample_x_t, outsample_x_t], axis=2)

        x_t = x_t * self.weight # Element-wise multiplication, broadcasted on b and t. Weights used in L1 regularization
        x_t = self.wavenet(x_t)[:]

        backcast_basis = x_t[:,:, :input_size]
        forecast_basis = x_t[:,:, input_size:]

        return backcast_basis, forecast_basis

    def construct(self, theta: ms.Tensor, insample_x_t: ms.Tensor, outsample_x_t: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        a = ms.ops.expand_dims(theta[:, cut_point:], 1)
        backcast = ms.ops.squeeze(ms.ops.matmul(a, backcast_basis))
        b = ms.ops.expand_dims(theta[:, :cut_point], 1)
        forecast = ms.ops.squeeze(ms.ops.matmul(b, forecast_basis))
        return backcast, forecast

class ExogenousBasisTCN(nn.Cell):
    def __init__(self, out_features, in_features, num_levels = 4, kernel_size=2, dropout_prob=0):
        super().__init__()
        n_channels = num_levels * [out_features]
        self.tcn = TemporalConvNet(num_inputs=in_features, num_channels=n_channels, kernel_size=kernel_size, dropout=dropout_prob)

    def transform(self, insample_x_t, outsample_x_t):
        input_size = insample_x_t.shape[2]

        x_t = ms.ops.concat([insample_x_t, outsample_x_t], axis=2)
        x_t = self.tcn(x_t)[:]
        backcast_basis = x_t[:,:, :input_size]
        forecast_basis = x_t[:,:, input_size:]

        return backcast_basis, forecast_basis

    def construct(self, theta: ms.Tensor, insample_x_t: ms.Tensor, outsample_x_t: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)
        cut_point = forecast_basis.shape[1]
        a = ms.ops.expand_dims(theta[:, cut_point:], 1)
        backcast = ms.ops.squeeze(ms.ops.matmul(a, backcast_basis))
        b = ms.ops.expand_dims(theta[:, :cut_point], 1)
        forecast = ms.ops.squeeze(ms.ops.matmul(b, forecast_basis))

        return backcast, forecast

def init_weights(module, initialization):
    if type(module) == ms.nn.Dense:
        if initialization == 'he_uniform':
            ms.common.initializer.HeUniform(module.weight)
        elif initialization == 'he_normal':
            ms.common.initializer.HeNormal(module.weight)
        elif initialization == 'glorot_uniform':
            ms.common.initializer.XavierUniform(module.weight)
        elif initialization == 'lecun_normal':
            pass
        else:
            assert 1<0, f'Initialization {initialization} not found'

class Nbeats(object):
    """
    Future documentation
    """
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    IDENTITY_BLOCK = 'identity'

    def __init__(self,
                 input_size_multiplier,
                 output_size,
                 shared_weights,
                 activation,
                 initialization,
                 stack_types,
                 n_blocks,
                 n_layers,
                 n_hidden,
                 n_harmonics,
                 n_polynomials,
                 exogenous_n_channels,
                 include_var_dict,
                 t_cols,
                 batch_normalization,
                 dropout_prob_theta,
                 dropout_prob_exogenous,
                 x_s_n_hidden,
                 learning_rate,
                 lr_decay,
                 n_lr_decay_steps,
                 weight_decay,
                 l1_theta,
                 n_iterations,
                 early_stopping,
                 loss,
                 loss_hypar,
                 val_loss,
                 random_seed,
                 seasonality,
                 device=None,
                 distribute=False,
                 rank_id=0):
        super(Nbeats, self).__init__()
        """
        N-BEATS model.
        Parameters
        ----------
        input_size_multiplier: int
            Multiplier to get insample size.
            Insample size = input_size_multiplier * output_size
        output_size: int
            Forecast horizon.
        shared_weights: bool
            If True, repeats first block.
        activation: str
            Activation function.
            An item from ['relu', 'softplus', 'tanh', 'selu', 'lrelu', 'prelu', 'sigmoid'].
        initialization: str
            Initialization function.
            An item from ['orthogonal', 'he_uniform', 'glorot_uniform', 'glorot_normal', 'lecun_normal'].
        stack_types: List[str]
            List of stack types.
            Subset from ['seasonality', 'trend', 'identity', 'exogenous', 'exogenous_tcn', 'exogenous_wavenet'].
        n_blocks: List[int]
            Number of blocks for each stack type.
            Note that len(n_blocks) = len(stack_types).
        n_layers: List[int]
            Number of layers for each stack type.
            Note that len(n_layers) = len(stack_types).
        n_hidden: List[List[int]]
            Structure of hidden layers for each stack type.
            Each internal list should contain the number of units of each hidden layer.
            Note that len(n_hidden) = len(stack_types).
        n_harmonics: List[int]
            Number of harmonic terms for each stack type.
            Note that len(n_harmonics) = len(stack_types).
        n_polynomials: List[int]
            Number of polynomial terms for each stack type.
            Note that len(n_polynomials) = len(stack_types).
        exogenous_n_channels:
            Exogenous channels for non-interpretable exogenous basis.
        include_var_dict: Dict[str, List[int]]
            Exogenous terms to add.
        t_cols: List
            Ordered list of ['y'] + X_cols + ['available_mask', 'sample_mask'].
            Can be taken from the dataset.
        batch_normalization: bool
            Whether perform batch normalization.
        dropout_prob_theta: float
            Float between (0, 1).
            Dropout for Nbeats basis.
        dropout_prob_exogenous: float
            Float between (0, 1).
            Dropout for exogenous basis.
        x_s_n_hidden: int
            Number of encoded static features to calculate.
        learning_rate: float
            Learning rate between (0, 1).
        lr_decay: float
            Decreasing multiplier for the learning rate.
        n_lr_decay_steps: int
            Period for each lerning rate decay.
        weight_decay: float
            L2 penalty for optimizer.
        l1_theta: float
            L1 regularization for the loss function.
        n_iterations: int
            Number of training steps.
        early_stopping: int
            Early stopping interations.
        loss: str
            Loss to optimize.
            An item from ['MAPE', 'MASE', 'SMAPE', 'MSE', 'MAE', 'PINBALL'].
        loss_hypar:
            Hyperparameter for chosen loss.
        val_loss:
            Validation loss.
            An item from ['MAPE', 'MASE', 'SMAPE', 'RMSE', 'MAE', 'PINBALL'].
        random_seed: int
            random_seed for pseudo random mindspore initializer and
            numpy random generator.
        seasonality: int
            Time series seasonality.
            Usually 7 for daily data, 12 for monthly data and 4 for weekly data.
        device: Optional[str]
            If None checks 'cuda' availability.
            An item from ['cuda', 'cpu'].
        """

        if activation == 'selu': initialization = 'lecun_normal'

        #------------------------ Model Attributes ------------------------#
        # Architecture parameters
        self.input_size = int(input_size_multiplier*output_size)
        self.output_size = output_size
        self.shared_weights = shared_weights
        self.activation = activation
        self.initialization = initialization
        self.stack_types = stack_types
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_harmonics = n_harmonics
        self.n_polynomials = n_polynomials
        self.exogenous_n_channels = exogenous_n_channels

        # Regularization and optimization parameters
        self.batch_normalization = batch_normalization
        self.dropout_prob_theta = dropout_prob_theta
        self.dropout_prob_exogenous = dropout_prob_exogenous
        self.x_s_n_hidden = x_s_n_hidden
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.n_lr_decay_steps = n_lr_decay_steps
        self.weight_decay = weight_decay
        self.n_iterations = n_iterations
        self.early_stopping = early_stopping
        self.loss = loss
        self.loss_hypar = loss_hypar
        self.val_loss = val_loss
        self.l1_theta = l1_theta
        self.l1_conv = 1e-3 # Not a hyperparameter
        self.random_seed = random_seed

        # Data parameters
        self.seasonality = seasonality
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols

        self._is_instantiated = False
        self.distribute = distribute
        self.rank_id = rank_id

    def create_stack(self):
        if self.include_var_dict is not None:
            x_t_n_inputs = self.output_size * int(sum([len(x) for x in self.include_var_dict.values()]))

            # Correction because week_day only adds 1 no output_size
            if len(self.include_var_dict['week_day'])>0:
                x_t_n_inputs = x_t_n_inputs - self.output_size + 1
        else:
            x_t_n_inputs = self.input_size

        #------------------------ Model Definition ------------------------#
        block_list = []
        self.blocks_regularizer = []
        for i in range(len(self.stack_types)):
            for block_id in range(self.n_blocks[i]):

                # Batch norm only on first block
                if (len(block_list)==0) and (self.batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Dummy of regularizer in block. Override with 1 if exogenous_block
                self.blocks_regularizer += [0]

                # Shared weights
                if self.shared_weights and block_id>0:
                    nbeats_block = block_list[-1]
                else:
                    if self.stack_types[i] == 'seasonality':
                        nbeats_block = NBeatsBlock(x_t_n_inputs = x_t_n_inputs,
                                                   x_s_n_inputs = self.n_x_s,
                                                   x_s_n_hidden= self.x_s_n_hidden,
                                                   theta_n_dim=4 * int(
                                                        np.ceil(self.n_harmonics / 2 * self.output_size) - (self.n_harmonics - 1)),
                                                   basis=SeasonalityBasis(harmonics=self.n_harmonics,
                                                                          backcast_size=self.input_size,
                                                                          forecast_size=self.output_size),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'trend':
                        nbeats_block = NBeatsBlock(x_t_n_inputs = x_t_n_inputs,
                                                   x_s_n_inputs = self.n_x_s,
                                                   x_s_n_hidden= self.x_s_n_hidden,
                                                   theta_n_dim=2 * (self.n_polynomials + 1),
                                                   basis=TrendBasis(degree_of_polynomial=self.n_polynomials,
                                                                            backcast_size=self.input_size,
                                                                            forecast_size=self.output_size),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'identity':
                        nbeats_block = NBeatsBlock(x_t_n_inputs = x_t_n_inputs,
                                                   x_s_n_inputs = self.n_x_s,
                                                   x_s_n_hidden= self.x_s_n_hidden,
                                                   theta_n_dim=self.input_size + self.output_size,
                                                   basis=IdentityBasis(backcast_size=self.input_size,
                                                                       forecast_size=self.output_size),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'exogenous':
                        nbeats_block = NBeatsBlock(x_t_n_inputs = x_t_n_inputs,
                                                   x_s_n_inputs = self.n_x_s,
                                                   x_s_n_hidden= self.x_s_n_hidden,
                                                   theta_n_dim=2*self.n_x_t,
                                                   basis=ExogenousBasisInterpretable(),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'exogenous_tcn':
                        nbeats_block = NBeatsBlock(x_t_n_inputs = x_t_n_inputs,
                                                   x_s_n_inputs = self.n_x_s,
                                                   x_s_n_hidden = self.x_s_n_hidden,
                                                   theta_n_dim = 2*(self.exogenous_n_channels),
                                                   basis= ExogenousBasisTCN(self.exogenous_n_channels, self.n_x_t),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'exogenous_wavenet':
                        nbeats_block = NBeatsBlock(x_t_n_inputs = x_t_n_inputs,
                                                   x_s_n_inputs = self.n_x_s,
                                                   x_s_n_hidden= self.x_s_n_hidden,
                                                   theta_n_dim=2*(self.exogenous_n_channels),
                                                   basis=ExogenousBasisWavenet(self.exogenous_n_channels, self.n_x_t),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                        self.blocks_regularizer[-1] = 1
                    else:
                        assert 1<0, f'Block type not found!'
                # Select type of evaluation and apply it to all layers of block
                init_function = partial(init_weights, initialization=self.initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list

    def __loss_fn(self, loss_name: str):
        def loss(x, loss_hypar, forecast, target, mask):
            if loss_name == 'MAE':
                return MAELoss(y=target, y_hat=forecast, mask=mask) + \
                       self.loss_l1_conv_layers() + self.loss_l1_theta()
            else:
                raise Exception(f'Unknown loss function: {loss_name}')
        return loss

    def __val_loss_fn(self, loss_name='MAE'):
        def loss(forecast, target, weights):
            if loss_name == 'MAE':
                return mae(y=target, y_hat=forecast, weights=weights)
            else:
                raise Exception(f'Unknown loss function: {loss_name}')
        return loss

    def loss_l1_conv_layers(self):
        loss_l1 = 0
        for i, indicator in enumerate(self.blocks_regularizer):
            if indicator:
                loss_l1 += self.l1_conv * ms.ops.sum(ms.ops.abs(self.model.blocks[i].basis.weight))
        return loss_l1

    def loss_l1_theta(self):
        loss_l1 = 0
        for block in self.model.blocks:
            for layer in block.cells_and_names():
                if isinstance(layer, ms.nn.Dense):
                    loss_l1 += self.l1_theta * layer.weight.abs().sum()
        return loss_l1

    def to_tensor(self, x):
        if 0 in x.shape:
            return None
        if type(x) is np.ndarray:
            return ms.Tensor(x, dtype=ms.float32)
        x = ms.ops.Cast()(x, ms.float32)
        return x

    def test(self, train_ts_loader, val_ts_loader=None, ckpt_path=None):
        assert (self.input_size)==train_ts_loader.input_size, \
            f'model input_size {self.input_size} data input_size {train_ts_loader.input_size}'

        # Random Seeds (model initialization)
        ms.set_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Attributes of ts_dataset
        self.n_x_t, self.n_x_s = train_ts_loader.get_n_variables()

        # Instantiate model
        if not self._is_instantiated:
            block_list = self.create_stack()
            self.model = NBeats(ms.nn.SequentialCell(block_list))
            self._is_instantiated = True
        print('>>>Loading from : {}>>'.format(ckpt_path))
        ms.load_param_into_net(self.model, ms.load_checkpoint(ckpt_path))
        print('>>>>>>>Load Succeed!>>>>>>>>>>>>>>>>>>>>>>>>>>')
        self.final_outsample_loss = self.evaluate_performance(ts_loader=val_ts_loader, validation_loss_fn=self.__val_loss_fn(self.val_loss))
        print("Test Loss: ", self.final_outsample_loss)
    
    def fit(self, train_ts_loader, val_ts_loader=None, n_iterations=None, verbose=True, eval_steps=1):

        assert (self.input_size)==train_ts_loader.input_size, \
            f'model input_size {self.input_size} data input_size {train_ts_loader.input_size}'

        # Random Seeds (model initialization)
        ms.set_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Attributes of ts_dataset
        self.n_x_t, self.n_x_s = train_ts_loader.get_n_variables()

        # Instantiate model
        if not self._is_instantiated:
            block_list = self.create_stack()
            self.model = NBeats(ms.nn.SequentialCell(block_list))
            self._is_instantiated = True
        
        # Overwrite n_iterations and train datasets
        if n_iterations is None:
            n_iterations = self.n_iterations

        lr_decay_steps = n_iterations // self.n_lr_decay_steps
        if lr_decay_steps == 0:
            lr_decay_steps = 1
        optimizer = ms.nn.Adam(params=self.model.trainable_params(), learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        training_loss_fn = self.__loss_fn(self.loss)
        validation_loss_fn = self.__val_loss_fn(self.val_loss)
    
        print('\n')
        print('='*30+' Start fitting '+'='*30)

        start = time.time()
        self.trajectories = {'iteration':[],'train_loss':[], 'val_loss':[]}
        self.final_insample_loss = None
        self.final_outsample_loss = None

        # Training Loop
        early_stopping_counter = 0
        best_val_loss = np.inf
        ms.save_checkpoint(self.model, f"./checkpoints/train_ckpt/Nbeatsx_best_{self.rank_id}.ckpt")
        break_flag = False
        iteration = 0
        epoch = 0

        def forward_fn(s_matrix, insample_y, insample_x, outsample_x, insample_mask):
            forecast = self.model(x_s=s_matrix, insample_y=insample_y,
                                        insample_x_t=insample_x, outsample_x_t=outsample_x,
                                        insample_mask=insample_mask)
            training_loss = training_loss_fn(x=insample_y, loss_hypar=self.loss_hypar, forecast=forecast,
                                                 target=outsample_y, mask=outsample_mask)
            return training_loss, forecast
        
        def clip_grad_norm(grads, max_norm=1.0, norm_type=2):
            total_norm = 0
            for p in grads:
                param_norm = np.linalg.norm(p.asnumpy().flatten(), ord=norm_type)
                # param_norm = ms.ops.flatten(p).norm(norm_type).asnumpy()
                total_norm += param_norm.item() ** norm_type
            total_norm = total_norm ** (1. / norm_type)
            clip_coef = max_norm / (total_norm + 1e-6)
            return min(clip_coef, 1)
        if self.distribute:
            mean = ms.context.get_auto_parallel_context("gradients_mean")
            degree = ms.context.get_auto_parallel_context("device_num")
            grad_reducer = ms.nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        
        while (iteration < n_iterations) and (not break_flag):
            epoch +=1
            for batch in iter(train_ts_loader):
                iteration += 1
                if (iteration > n_iterations) or (break_flag):
                    continue

                self.model.set_train(True)
                # Parse batch
                insample_y     = self.to_tensor(batch['insample_y'])
                insample_x     = self.to_tensor(batch['insample_x'])
                insample_mask  = self.to_tensor(batch['insample_mask'])
                outsample_x    = self.to_tensor(batch['outsample_x'])
                outsample_y    = self.to_tensor(batch['outsample_y'])
                outsample_mask = self.to_tensor(batch['outsample_mask'])
                s_matrix       = self.to_tensor(batch['s_matrix'])
                grad_fn = ms.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
                (loss, _), grads = grad_fn(s_matrix, insample_y, insample_x, outsample_x, insample_mask)
                if not np.isnan(float(loss)):
                    loss_scale = StaticLossScaler(1.0 / clip_grad_norm(grads))
                    grads = loss_scale.unscale(grads)
                    if self.distribute:
                        grads = grad_reducer(grads)
                    loss = ms.ops.depend(loss, optimizer(grads))
                else:
                    early_stopping_counter = self.early_stopping

                if (iteration % eval_steps == 0):
                    display_string = 'Step: {}, Time: {:03.3f}, Insample {}: {:.5f}'.format(iteration,
                                                                                            time.time()-start,
                                                                                            self.loss,
                                                                                            loss.asnumpy())
                    self.trajectories['iteration'].append(iteration)
                    self.trajectories['train_loss'].append(np.float(loss.asnumpy()))
                    Insample_loss = loss.asnumpy()

                    if val_ts_loader is not None:
                        loss = self.evaluate_performance(ts_loader=val_ts_loader,
                                                         validation_loss_fn=validation_loss_fn)
                        display_string += ", Outsample {}: {:.5f}".format(self.val_loss, loss)
                        self.trajectories['val_loss'].append(loss)

                        if self.early_stopping:
                            if loss < best_val_loss:
                                # Save current model if improves outsample loss
                                ms.save_checkpoint(self.model, f"./checkpoints/train_ckpt/Nbeatsx_best_{self.rank_id}.ckpt")
                                # best_state_dict = copy.deepcopy(self.model.state_dict())
                                best_insample_loss = Insample_loss
                                early_stopping_counter = 0
                                best_val_loss = loss
                            else:
                                early_stopping_counter += 1
                            if early_stopping_counter >= self.early_stopping:
                                break_flag = True

                    print(display_string)

                    self.model.set_train(True)

                if break_flag:
                    print('\n')
                    print(19*'-',' Stopped training by early stopping', 19*'-')
                    break

        #End of fitting
        if n_iterations > 0:
            # This is batch loss!
            print('='*30+'  End fitting  '+'='*30)
            self.final_insample_loss = best_insample_loss
            string = 'Step: Last, Time: {:03.3f}, Insample {}: {:.5f}'.format(time.time()-start,
                                                                            self.loss,
                                                                            self.final_insample_loss)
            if val_ts_loader is not None:
                ms.load_param_into_net(self.model, ms.load_checkpoint(f"./checkpoints/train_ckpt/Nbeatsx_best_{self.rank_id}.ckpt"))
                self.final_outsample_loss = self.evaluate_performance(ts_loader=val_ts_loader,
                                                                      validation_loss_fn=validation_loss_fn)
                string += ", Outsample {}: {:.5f}".format(self.val_loss, self.final_outsample_loss)
            print(string)

    def predict(self, ts_loader, X_test=None, return_decomposition=False):
        self.model.set_train(False)
        assert not ts_loader.shuffle, 'ts_loader must have shuffle as False.'

        forecasts = []
        block_forecasts = []
        outsample_ys = []
        outsample_masks = []
        for batch in iter(ts_loader):
            insample_y     = self.to_tensor(batch['insample_y'])
            insample_x     = self.to_tensor(batch['insample_x'])
            insample_mask  = self.to_tensor(batch['insample_mask'])
            outsample_x    = self.to_tensor(batch['outsample_x'])
            s_matrix       = self.to_tensor(batch['s_matrix'])

            forecast, block_forecast = self.model(insample_y=insample_y, insample_x_t=insample_x,
                                                    insample_mask=insample_mask, outsample_x_t=outsample_x,
                                                    x_s=s_matrix, return_decomposition=True) # always return decomposition
            forecasts.append(forecast.asnumpy())
            block_forecasts.append(block_forecast.asnumpy())
            outsample_ys.append(batch['outsample_y'].asnumpy())
            outsample_masks.append(batch['outsample_mask'].asnumpy())

        forecasts = np.vstack(forecasts)
        block_forecasts = np.vstack(block_forecasts)
        outsample_ys = np.vstack(outsample_ys)
        outsample_masks = np.vstack(outsample_masks)

        self.model.set_train(True)
        if return_decomposition:
            return outsample_ys, forecasts, block_forecasts, outsample_masks
        else:
            return outsample_ys, forecasts, outsample_masks

    def evaluate_performance(self, ts_loader, validation_loss_fn):
        self.model.set_train(False)

        target, forecast, outsample_mask = self.predict(ts_loader=ts_loader)

        complete_loss = validation_loss_fn(target=target, forecast=forecast, weights=outsample_mask)

        self.model.set_train(True)
        return complete_loss