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

import numpy as np
import mindspore
import mindspore.ops as F

class MultiLayerPerceptron(mindspore.nn.Cell):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(mindspore.nn.Dense(input_dim, embed_dim))
            layers.append(mindspore.nn.BatchNorm1d(embed_dim))
            layers.append(mindspore.nn.ReLU())
            layers.append(mindspore.nn.Dropout(keep_prob=1-dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(mindspore.nn.Dense(input_dim, 1))
        self.mlp = mindspore.nn.SequentialCell(*layers)
        for param in self.mlp.get_parameters():
            if param.name == 'mlp.8.weight':
                param.name = 'output_l.weight'
            if param.name == 'mlp.8.bias':
                param.name = 'output_l.bias'
    def construct(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class FeaturesEmbedding(mindspore.nn.Cell):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = mindspore.nn.Embedding(int(sum(field_dims)), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        mindspore.common.initializer.XavierUniform(self.embedding.embedding_table.data)

    def construct(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        offset = mindspore.Tensor.from_numpy(self.offsets)
        expand_dims = mindspore.ops.ExpandDims()
        offset = expand_dims(offset, 0)
        x = x + offset
        return self.embedding(x)