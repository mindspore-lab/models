# Copyright 2021 Xidian University
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
"""LeNet model for ADDA."""

import mindspore.nn as nn


class LeNetEncoder(nn.Cell):
    def __init__(self):
        super(LeNetEncoder, self).__init__()
        conv1 = nn.Conv2d(1, 20, 5, pad_mode='valid')
        maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(20, 50, 5, pad_mode='valid')
        dropout = nn.Dropout()
        maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        relu2 = nn.ReLU()
        self.encoder = nn.SequentialCell([conv1, maxpool1, relu1, conv2, dropout, maxpool2, relu2])
        self.fc1 = nn.Dense(50 * 4 * 4, 500)

    def construct(self, x):
        conv_out = self.encoder(x)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat


class LeNetClassifier(nn.Cell):
    """LeNet classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Dense(500, 10)

    def construct(self, feat):
        """Forward the LeNet classifier."""

        out = self.dropout(self.relu(feat))
        out = self.fc2(out)
        return out
