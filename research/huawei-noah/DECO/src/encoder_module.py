# Copyright 2022 Huawei Technologies Co., Ltd
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

"""
DECO Encoder class. Built with ConvNeXt blocks.
"""

import numpy as np
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.common import initializer as init
from src.drop_path import DropPath2D


class Block(nn.Cell):
    r""" ConvNeXt Block. 
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, group=dim, has_bias=True)  
        self.norm = ConvNextLayerNorm((dim,), epsilon=1e-6)
        self.pwconv1 = nn.Dense(dim, 4 * dim)  
        self.act = nn.GELU()
        self.pwconv2 = nn.Dense(4 * dim, dim)
        self.gamma = Parameter(Tensor(layer_scale_init_value * np.ones((dim)), dtype=mstype.float32),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath2D(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x):
        """Block construct"""
        downsample = x
        x = self.dwconv(x)
        x = ops.Transpose()(x, (0, 2, 3, 1))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = ops.Transpose()(x, (0, 3, 1, 2))
        x = downsample + self.drop_path(x)
        return x


class ConvNeXt(nn.Cell):
    r""" ConvNeXt
    Args:
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, 
                 depths=[2, 6, 2], dims=[120, 240, 480], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, 
                 ):
        super().__init__()

        self.downsample_layers = nn.CellList()  
        for i in range(len(depths)-1):
            downsample_layer = nn.SequentialCell(
                ConvNextLayerNorm((dims[i],), epsilon=1e-6, norm_axis=1),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=1, has_bias=True),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.CellList()  
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            blocks = []
            for j in range(depths[i]):
                blocks.append(Block(dim=dims[i], drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=layer_scale_init_value))
            stage = nn.SequentialCell(blocks)
            self.stages.append(stage)
            cur += depths[i]

        self.norm = ConvNextLayerNorm((dims[-1],), epsilon=1e-6) 
        self.init_weights()


    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv2d)):
                cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                cell.bias.set_data(init.initializer(init.Zero(),
                                                    cell.bias.shape,
                                                    cell.bias.dtype))

    def forward_features(self, x):
        for i in range(2):
            x = self.stages[i](x)
            x = self.downsample_layers[i](x)
        x = self.stages[2](x)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        return x


class ConvNextLayerNorm(nn.LayerNorm):
    """ConvNextLayerNorm"""
    def __init__(self, normalized_shape, epsilon, norm_axis=-1):
        super(ConvNextLayerNorm, self).__init__(normalized_shape=normalized_shape, epsilon=epsilon)
        assert norm_axis in (-1, 1), "ConvNextLayerNorm's norm_axis must be 1 or -1."
        self.norm_axis = norm_axis

    def construct(self, input_x):
        if self.norm_axis == -1:
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
        else:
            input_x = ops.Transpose()(input_x, (0, 2, 3, 1))
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
            y = ops.Transpose()(y, (0, 3, 1, 2))
        return y

