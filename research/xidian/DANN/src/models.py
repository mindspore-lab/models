# Copyright 2023 Xidian University
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
import numpy as np
import mindspore as ms
from mindspore.common import initializer as init
import mindspore.ops as ops

class Backbone(nn.Cell):
    def __init__(self):
        super(Backbone, self).__init__()
        #backbone
        f_conv1 = nn.Conv2d(3, 64, kernel_size=5,pad_mode='valid', has_bias=True,weight_init=init.HeUniform())
        f_bn1 = nn.BatchNorm2d(64,beta_init='ones')
        f_pool1 = nn.MaxPool2d(kernel_size=2,pad_mode='valid',stride=2)
        f_relu1 = nn.ReLU()
        f_conv2 = nn.Conv2d(64, 50, kernel_size=5,pad_mode='valid', has_bias=True,weight_init=init.HeUniform())
        f_bn2 = nn.BatchNorm2d(50)
        f_drop1 = nn.Dropout()
        f_pool2 = nn.MaxPool2d(kernel_size=2,pad_mode='valid',stride=2)
        f_relu2 = nn.ReLU()
        self.feature = nn.SequentialCell([f_conv1, f_bn1, f_pool1, f_relu1, f_conv2, f_bn2, f_drop1, f_pool2, f_relu2])

    def construct(self, x):
        y = ms.Tensor(np.ones((64, 3,28,28)))
        x = x.expand_as(y)
        feature = self.feature(x)
        return feature.view(-1, 50 * 4 * 4)

class Class_classifier(nn.Cell):
    def __init__(self):
        super(Class_classifier, self).__init__()
        c_fc1 = nn.Dense(50 * 4 * 4, 100, weight_init=init.XavierUniform()).to_float(ms.float16)
        c_bn1 = nn.BatchNorm1d(100)
        c_relu1 = nn.ReLU()
        c_drop1 = nn.Dropout()
        c_fc2 = nn.Dense(100, 100, weight_init=init.XavierUniform()).to_float(ms.float16)
        c_bn2 = nn.BatchNorm1d(100)
        c_relu2 = nn.ReLU()
        c_fc3 = nn.Dense(100, 10, weight_init=init.XavierUniform()).to_float(ms.float16)
        c_softmax = nn.LogSoftmax(axis=1)
        self.class_classifier = nn.SequentialCell([c_fc1, c_bn1, c_relu1, c_drop1, c_fc2, c_bn2, c_relu2, c_fc3, c_softmax])

    def construct(self, x):
        x = ops.Cast()(x, ms.float16)
        output = self.class_classifier(x)
        return output

class Domain_classifier(nn. Cell):
    def __init__(self):
        super(Domain_classifier, self).__init__()
        d_fc1 = nn.Dense(50 * 4 * 4, 100, weight_init=init.XavierUniform()).to_float(ms.float16)
        d_relu1 = nn.ReLU()
        d_drop1 = nn.Dropout()
        d_fc2 = nn.Dense(100, 2, weight_init=init.XavierUniform()).to_float(ms.float16)
        d_softmax = nn.LogSoftmax(axis=1)
        self.domain_classifier = nn.SequentialCell([d_fc1, d_relu1, d_drop1, d_fc2, d_softmax])

    def construct(self, x):
        x = ops.Cast()(x, ms.float16)
        output = self.domain_classifier(x)
        return output

