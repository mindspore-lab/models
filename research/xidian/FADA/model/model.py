# Copyright 2024 Xidian University
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
from mindspore import nn


class DCD(nn.Cell):
    def __init__(self, h_features=64, input_features=128):
        super().__init__()
        self.softmax = nn.Softmax(axis=1)
        self.fc1 = nn.Dense(input_features, h_features,
                            weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(h_features, h_features,
                            weight_init="normal", bias_init="zeros")
        self.fc3 = nn.Dense(
            h_features, 4, weight_init="normal", bias_init="zeros")

    def construct(self, x):
        case = self.fc1(x)
        case = self.relu(case)
        case = self.fc2(case)
        case = self.fc3(case)

        return self.softmax(case)


class Classifier(nn.Cell):
    def __init__(self, input_features=64):
        super().__init__()
        self.softmax = nn.Softmax(axis=1)
        self.fc = nn.Dense(input_features, 10)

    def construct(self, x):
        logits = self.fc(x)
        return self.softmax(logits)


class Encoder(nn.Cell):
    def __init__(self):
        super().__init__()

        self.conv = nn.SequentialCell(
            nn.Conv2d(1, 6, 5, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, 5, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.SequentialCell(
            nn.Dense(256, 120, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(120, 84, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(84, 64, weight_init="normal", bias_init="zeros"),
        )

    def construct(self, x):
        logits = self.conv(x)
        logits = logits.view(logits.shape[0], -1)
        out = self.fc(logits)
        return out
