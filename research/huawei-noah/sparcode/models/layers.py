# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import mindspore.nn as nn


class MLP_Layer(nn.Cell):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_units=[],
        hidden_activations="ReLU",
        output_activation=None,
        dropout_rates=[],
        batch_norm=False,
        use_bias=True,
    ):
        super(MLP_Layer, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(
                nn.Dense(hidden_units[idx], hidden_units[idx + 1], bias_init=use_bias)
            )
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(
                nn.Dense(hidden_units[-1], output_dim, bias_init=use_bias)
            )
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.dnn = nn.SequentialCell(*dense_layers)  # * used to unpack list

    def construct(self, inputs):
        return self.dnn(inputs)


#
def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            return getattr(nn, activation)()
    else:
        return activation


class SELayer(nn.Cell):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.SequentialCell(
            nn.Dense(channel, channel // reduction),
            nn.ReLU(),
            nn.Dense(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def construct(self, x):
        b, t, _ = x.size()
        y = self.avg_pool(x).view(b, t)
        y = self.fc(y).view(b, t, 1)
        return x * y.expand_as(x)
