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
import mindspore
import mindspore.nn as nn
from mindspore import Tensor,context


context.set_context(mode=context.PYNATIVE_MODE)


class FCDiscriminator(nn.Cell):

    def __init__(self, num_classes=19, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_classes, out_channels=ndf, kernel_size=4, stride=2, padding=1,
                               # weight_init=init.Normal(0.01,0),
                               pad_mode='pad')
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1,
                               # weight_init=init.Normal(0.01, 0),
                               pad_mode='pad')
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1,
                               # weight_init=init.Normal(0.01, 0),
                               pad_mode='pad')
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1,
                               # weight_init=init.Normal(0.01, 0),
                               pad_mode='pad')
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1,
                                    # weight_init=init.Normal(0.01, 0),
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


class FCDiscriminator_Local(nn.Cell):

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator_Local, self).__init__()

        self.conv1 = nn.Conv2d(num_classes + 2048, ndf, kernel_size=4, stride=2, padding=1, pad_mode='pad')
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, pad_mode='pad')
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, pad_mode='pad')
        self.classifier = nn.Conv2d(ndf*4, 1, kernel_size=4, stride=2, padding=1, pad_mode='pad')

        self.leaky_relu = nn.LeakyReLU(alpha=0.2)
        self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        #self.sigmoid = nn.Sigmoid()


    def construct(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = self.up_sample(x)
        #x = self.sigmoid(x) 

        return x

if __name__ == '__main__':
    import numpy as np

    input = np.random.random((1, 19, 64, 64))
    input = mindspore.Tensor(input, mindspore.float32)
    model=FCDiscriminator(num_classes=19)
    out=model(input)
