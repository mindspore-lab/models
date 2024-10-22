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

""" Classifier head and layer factory

Hacked together by / Copyright 2020 Ross Wightman
"""

import mindspore.nn as nn

from .adaptive_avgmax_pool import SelectAdaptivePool2d


def create_classifier(num_features, num_classes, pool_type='avg', use_conv=False):
    flatten = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=flatten)
    num_pooled_features = num_features * global_pool.feat_mult()
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_pooled_features, num_classes, 1, has_bias=True, pad_mode='valid')
    else:
        # NOTE: using my Linear wrapper that fixes AMP + torchscript casting issue
        fc = nn.Dense(num_pooled_features, num_classes, has_bias=True)
    return global_pool, fc


class ClassifierHead(nn.Cell):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0.):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, self.fc = create_classifier(in_chs, num_classes, pool_type=pool_type)
        self.dropout = nn.Dropout(1.0 - float(self.drop_rate))

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = self.global_pool(x)
        if self.drop_rate:
            x = self.dropout(x)
        x = self.fc(x)
        return x
