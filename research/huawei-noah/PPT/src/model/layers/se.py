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
#
# This file has been derived from the https://github.com/huggingface/pytorch-image-models
# repository and modified.
# ============================================================================


import mindspore.nn as nn



class SEModule(nn.Cell):

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, min_channels=8, reduction_channels=None,
                 gate_layer=nn.Sigmoid()):
        super(SEModule, self).__init__()
        reduction_channels = reduction_channels or max(channels // reduction, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, has_bias=True, pad_mode='valid')
        self.act = act_layer()#inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, has_bias=True, pad_mode='valid')
        self.gate = gate_layer

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x_se = x.mean((2, 3), keep_dims=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
