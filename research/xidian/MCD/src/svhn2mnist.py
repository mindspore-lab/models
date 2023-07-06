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

from .var_init import default_recurisive_init


class Feature(nn.Cell):
    def __init__(self):
        super(Feature, self).__init__(auto_prefix=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, pad_mode='pad', has_bias=True)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, pad_mode='pad', has_bias=True)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, pad_mode='pad', has_bias=True)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.fc1 = nn.Dense(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072, momentum=0.1)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.dropout = nn.Dropout()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.bn1(self.conv1(x))))
        x = self.max_pool2d(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], 8192)
        x = self.relu(self.bn1_fc(self.fc1(x)))
        x = self.dropout(x)
        return x


class Predictor(nn.Cell):
    def __init__(self):
        super(Predictor, self).__init__(auto_prefix=True)
        self.fc2 = nn.Dense(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048, momentum=0.1)
        self.fc3 = nn.Dense(2048, 10)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__(auto_prefix=True)
        self.G = Feature()
        self.C1 = Predictor()
        self.C2 = Predictor()

        default_recurisive_init(self.G)
        default_recurisive_init(self.C1)
        default_recurisive_init(self.C2)

    def construct(self, img):
        feat = self.G(img)
        output1 = self.C1(feat)
        output2 = self.C2(feat)
        return feat, output1, output2
