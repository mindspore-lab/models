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
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.initializer import Normal


class Bottleneck(nn.Cell):
    """build bottleneck module"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, freeze_bn_affine=True, use_batch_statistics=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
                               has_bias=False, weight_init=Normal(0.01))
        self.bn1 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)

        if freeze_bn_affine:
            for i in self.bn1.parameters_dict().values():
                i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, has_bias=False, dilation=dilation,
                               weight_init=Normal(0.01), pad_mode="pad")
        self.bn2 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)

        if freeze_bn_affine:
            for i in self.bn2.parameters_dict().values():
                i.requires_grad = False

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False, stride=1, weight_init=Normal(0.01))
        self.bn3 = nn.BatchNorm2d(planes * 4, use_batch_statistics=use_batch_statistics)

        if freeze_bn_affine:
            for i in self.bn3.parameters_dict().values():
                i.requires_grad = False

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.add = P.Add()

    def construct(self, x):
        """construct bottleneck module"""
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.add(out, residual)
        out = self.relu(out)
        return out


class Resnet(nn.Cell):
    def __init__(self, block, layers, replace_stride_with_dilation=None, freeze_bn_affine=True, use_batch_statistics=False):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               has_bias=False, pad_mode="pad")
        self.bn1 = nn.BatchNorm2d(self.inplanes, use_batch_statistics=use_batch_statistics)

        self.freeze_bn_affine = freeze_bn_affine
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        
        if self.freeze_bn_affine:
            for i in self.bn1.parameters_dict().values():
                i.requires_grad = False

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0], use_batch_statistics=use_batch_statistics)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, 
                                       dilate=replace_stride_with_dilation[0], use_batch_statistics=use_batch_statistics)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, 
                                       dilate=replace_stride_with_dilation[1], use_batch_statistics=use_batch_statistics)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, 
                                       dilate=replace_stride_with_dilation[2], use_batch_statistics=use_batch_statistics)

        if freeze_bn_affine:
            self.freeze_bn()

    def freeze_bn(self):
        """Freeze batch norms."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.BatchNorm2d):
                cell.beta.requires_grad = False
                cell.gamma.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_batch_statistics=False):
        """define layers"""
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False, weight_init=Normal(0.01)),
                nn.BatchNorm2d(planes * block.expansion, use_batch_statistics=use_batch_statistics)])

            if self.freeze_bn_affine:
                for i in downsample._cells['1'].parameters_dict().values():
                    i.requires_grad = False

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=previous_dilation, downsample=downsample,
                  freeze_bn_affine=self.freeze_bn_affine))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.dilation,
                                freeze_bn_affine=self.freeze_bn_affine, use_batch_statistics=use_batch_statistics))

        return nn.SequentialCell(layers)

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def Resnet_101(cfg, pretrained=True, replace_stride_with_dilation=[False, True, True], freeze_bn_affine=True, freeze_bn=True):
    
    use_batch_statistics = not freeze_bn

    resnet101 = Resnet(Bottleneck, [3, 4, 23, 3], 
                       replace_stride_with_dilation=replace_stride_with_dilation, freeze_bn_affine=freeze_bn_affine,
                        use_batch_statistics=use_batch_statistics)

    if pretrained:
        param_dict = load_checkpoint(cfg.MODEL.RES_PRETRAINED)
        trans_param_dict = {}
        for key, val in param_dict.items():
            key = key.replace("down_sample_layer", "downsample")
            trans_param_dict[f"network.resnet.{key}"] = val
        load_param_into_net(resnet101, trans_param_dict)
        print('load_model {} success'.format(cfg.MODEL.RES_PRETRAINED))

    return resnet101