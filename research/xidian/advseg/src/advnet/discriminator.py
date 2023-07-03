# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, context

context.set_context(mode=context.PYNATIVE_MODE)


class FCDiscriminator(nn.Cell):

    def __init__(self, num_classes=19, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_classes, out_channels=ndf, kernel_size=4, stride=2, padding=1,
                               pad_mode='pad')
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1,
                               pad_mode='pad')
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1,
                               pad_mode='pad')
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1,
                               pad_mode='pad')
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1,
                                    pad_mode='pad')

        self.leaky_relu = nn.LeakyReLU(alpha=0.2)

    def construct(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


class train_D_with_data(nn.Cell):
    def __init__(self, network, num_classes=19):
        super(train_D_with_data, self).__init__()
        self.num_classes = num_classes
        self.network = network

    def construct(self):
        out = self.network(self.data)
        return out


if __name__ == '__main__':
    import numpy as np

    np.random.seed(100)
    class_num = 19
    temp1 = Tensor(np.random.randint(0, 255, (1, class_num, 512, 512)), mindspore.float32)
    model = FCDiscriminator(num_classes=class_num)
    for param in model.get_parameters():
        print(param.name, param.asnumpy().mean(), param.asnumpy().std())
    out = model(temp1)
    print(out)
