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

"""Patch embedding layer """

import mindspore as ms
import mindspore.nn as nn

def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    return 1


class SelectAdaptivePool2d(nn.Cell):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='avg', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''
        self.flatten = flatten
        self.pool = ms.ops.AdaptiveAvgPool2D(output_size)

    def is_identity(self):
        return self.pool_type == ''

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = self.pool(x)
        if self.flatten:
            x = ms.nn.Flatten()(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'pool_type=' + self.pool_type \
            + ', flatten=' + str(self.flatten) + ')'
