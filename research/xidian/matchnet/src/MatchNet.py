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
"""MatchNet"""
from mindspore import nn
from mindspore import ops

from .var_init import default_recurisive_init


class FeatureNet(nn.Cell):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.features = nn.SequentialCell([
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=7, padding=3, pad_mode='pad', stride=1,
                      has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same"),
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=5, padding=2, pad_mode='pad', stride=1,
                      has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same"),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1, pad_mode='pad', stride=1,
                      has_bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, pad_mode='pad', stride=1,
                      has_bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1, pad_mode='pad', stride=1,
                      has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        ])

    def construct(self, x):
        return self.features(x)


class MetricNet(nn.Cell):
    def __init__(self):
        super(MetricNet, self).__init__()
        self.features = nn.SequentialCell([
            nn.Dense(in_channels=8192, out_channels=1024),
            nn.ReLU(),
            nn.Dense(in_channels=1024, out_channels=1024),
            nn.ReLU(),
            nn.Dense(in_channels=1024, out_channels=2)
        ])

    def construct(self, x):
        return self.features(x)


class MatchNet(nn.Cell):
    def __init__(self):
        super(MatchNet, self).__init__()
        self.input_ = FeatureNet()
        self.metric_network = MetricNet()
        default_recurisive_init(self.input_)
        default_recurisive_init(self.metric_network)

    def construct(self, x):
        feature1 = self.input_(x[0]).reshape((x[0].shape[0], -1))
        feature2 = self.input_(x[1]).reshape((x[1].shape[0], -1))
        features = ops.concat((feature1, feature2), axis=1)
        return self.metric_network(features)
