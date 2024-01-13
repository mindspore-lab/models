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

"""Implementation of the MLP layer."""


from functools import partial


import mindspore.nn as nn


class Mlp(nn.Cell):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=partial(nn.GELU, approximate=False),
            drop=0.
    ):
        """Mlp layer

        Parameters
        ----------
        in_features : int
            Input number of features
        hidden_features : Optional[int]
            Size of hidden layer
        out_features : Optional[int]
            Size of output layer
        act_layer : nn.Cell
            activation layer.
            NOTE: by default in original code it uses torch.nn.GELU,
            but we need to use partial(nn.GELU, approximate=False) because of
            difference between ms.nn.GELU(approximate=False) and
            ms.nn.GELU(approximate=True)

        drop : float
            Dropout probability
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(1.0 - drop)

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
