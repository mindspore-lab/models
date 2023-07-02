# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Discriminator model for ADDA."""

import mindspore.nn as nn


class Discriminator(nn.Cell):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Discriminator, self).__init__()
        dense1 = nn.Dense(input_dims, hidden_dims)
        relu1 = nn.ReLU()
        dense2 = nn.Dense(hidden_dims, hidden_dims)
        relu2 = nn.ReLU()
        dense3 = nn.Dense(hidden_dims, output_dims)
        logsoftmax = nn.LogSoftmax()
        self.layer = nn.SequentialCell([dense1, relu1, dense2, relu2, dense3, logsoftmax])

    def construct(self, x):
        out = self.layer(x)
        return out
