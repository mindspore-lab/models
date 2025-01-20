# Copyright 2021 Huawei Technologies Co., Ltd
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
'''Resnet'''
import os
from mindspore import nn
from typing import Type, Union, List, Optional
from mindvision.classification.models.blocks import ConvNormActivation
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.neck import GlobalAvgPooling
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
import mindspore as ms

class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None) -> None:
        super(ResidualBlock, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d

        self.conv1 = ConvNormActivation(in_channel, out_channel,
                                        kernel_size=1, norm=norm)
        self.conv2 = ConvNormActivation(out_channel, out_channel,
                                        kernel_size=3, stride=stride, norm=norm)
        self.conv3 = ConvNormActivation(out_channel, out_channel * self.expansion,
                                        kernel_size=1, norm=norm, activation=None)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        identity = x

        out = self.conv1(x) 
        out = self.conv2(out)
        out = self.conv3(out)

        if self.down_sample:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out

def make_layer(last_out_channel, block: Type[Union[ResidualBlock]],
               channel: int, block_nums: int, stride: int = 1):
    down_sample = None

    if stride != 1 or last_out_channel != channel * block.expansion:
        down_sample = ConvNormActivation(last_out_channel, channel * block.expansion,
                                         kernel_size=1, stride=stride, norm=nn.BatchNorm2d, activation=None)
    layers = []
    layers.append(block(last_out_channel, channel, stride=stride, down_sample=down_sample, norm=nn.BatchNorm2d))

    in_channel = channel * block.expansion
    for _ in range(1, block_nums):
        layers.append(block(in_channel, channel, norm=nn.BatchNorm2d))

    return nn.SequentialCell(layers)

class ResNet(nn.Cell):
    def __init__(self,block:Type[Union[ResidualBlock]],
                 layer_nums: List[int], norm:Optional[nn.Cell] = None) -> None:
        super(ResNet, self).__init__()

        if not norm:
            norm = nn.BatchNorm2d
            self.conv1 = ConvNormActivation(3, 64, kernel_size=7, stride=2, norm=norm)
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
            self.layer1 = make_layer(64, block, 64, layer_nums[0])
            self.layer2 = make_layer(64 * block.expansion, block, 128, layer_nums[1], stride=2)
            self.layer3 = make_layer(128 * block.expansion, block, 256, layer_nums[2], stride=2)
            self.layer4 = make_layer(256 * block.expansion, block, 512, layer_nums[3], stride=2)

    def construct(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def _resnet(arch:str, block:Type[Union[ResidualBlock]],
            layers: List[int],  pretrained: bool):

    backbone = ResNet(block, layers)
    neck = GlobalAvgPooling()
    model = BaseClassifier(backbone,neck)

    if pretrained:
        LoadPretrainedModel(model, model_urls[arch]).run()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dict = ms.load_checkpoint(os.path.join(current_dir,'LoadPretrainedModel/resnet50_224.ckpt'))
        ms.load_param_into_net(model,model_dict)
    return model
def resnet50(pretrained: bool = False):
    return _resnet('resnet50', ResidualBlock, [3, 4, 6, 3], pretrained)
