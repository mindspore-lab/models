from download import download
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
from mindvision.classification.models import resnet50
import matplotlib.pyplot as plt
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import nn
from mindspore.common import initializer
from mindspore import nn, ops
from mindspore import Parameter
import math


class FineB(nn.Cell):
    def __init__(self, num_class=200, pretrained=False):
        super(FineB, self).__init__()

        # 使用 ResNet-50 作为主干网络
        self.resnet = resnet50(pretrained=True)
        if pretrained:
            self.load_pretrained_resnet('./resnet50_ms_converted1.ckpt')

        self.features = nn.SequentialCell(*list(self.resnet.cells())[:-2])

        for param in self.features.trainable_params():
            param.requires_grad = True  # 设置为可训练（重要）

        #         # 优化后的分类层
        self.classifier = nn.SequentialCell(
            nn.Dense(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dense(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dense(128, num_class)
        )

        self.gap = ops.AdaptiveAvgPool2D(output_size=1)
        self.flatten = nn.Flatten()

        self.block = StripPooling(2048, (4, 2), nn.BatchNorm2d, {'mode': 'bilinear', 'align_corners': True})

    def load_pretrained_resnet(self, ckpt_file):
        pretrained_dict = ms.load_checkpoint(ckpt_file)
        model_dict = self.resnet.parameters_dict()
        for key in pretrained_dict:
            if key in model_dict:
                model_dict[key].set_data(pretrained_dict[key].data)

    def construct(self, x):
        # 前向：特征提取 -> 全局池化 -> 分类
        x = self.features(x)
        x = self.block(x)
        x = self.gap(x)  # [B, C, H, W] -> [B, C, 1, 1]
        x = self.flatten(x)  # [B, C, 1, 1] -> [B, C]
        out = self.classifier(x)
        return out


import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class StripPooling(nn.Cell):

    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        # 四种不同尺寸的条带池化层
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=pool_size[0])  # 水平条带
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=pool_size[1])  # 垂直条带
        self.pool3 = nn.AdaptiveAvgPool2d(output_size=(1, None))  # 整行
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(None, 1))  # 整列

        inter_channels = in_channels // 4
        # 初始 1×1 映射
        self.conv1_1 = nn.SequentialCell([
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, has_bias=False),
            norm_layer(inter_channels),
            nn.ReLU()
        ])
        self.conv1_2 = nn.SequentialCell([
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, has_bias=False),
            norm_layer(inter_channels),
            nn.ReLU()
        ])

        # 多分支卷积
        self.conv2_0 = nn.SequentialCell([
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
            norm_layer(inter_channels)
        ])
        self.conv2_1 = nn.SequentialCell([
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
            norm_layer(inter_channels)
        ])
        self.conv2_2 = nn.SequentialCell([
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
            norm_layer(inter_channels)
        ])
        self.conv2_3 = nn.SequentialCell([
            nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, 3), pad_mode='pad', padding=(0, 0, 1, 1),
                      has_bias=False),
            norm_layer(inter_channels)
        ])
        self.conv2_4 = nn.SequentialCell([
            nn.Conv2d(inter_channels, inter_channels, kernel_size=(3, 1), pad_mode='pad', padding=(1, 1, 0, 0),
                      has_bias=False),
            norm_layer(inter_channels)
        ])
        self.conv2_5 = nn.SequentialCell([
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
            norm_layer(inter_channels),
            nn.ReLU()
        ])
        self.conv2_6 = nn.SequentialCell([
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
            norm_layer(inter_channels),
            nn.ReLU()
        ])

        # 最后恢复通道
        self.conv3 = nn.SequentialCell([
            nn.Conv2d(inter_channels * 2, in_channels, kernel_size=1, has_bias=False),
            norm_layer(in_channels)
        ])

        # 插值参数存储
        self._up_kwargs = up_kwargs

    def construct(self, x):
        b, c, h, w = x.shape
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)

        # 多路池化 + 卷积 + 上采样
        x2_1 = self.conv2_0(x1)
        x2_2 = ops.interpolate(self.conv2_1(self.pool1(x1)), size=(h, w), **self._up_kwargs)
        x2_3 = ops.interpolate(self.conv2_2(self.pool2(x1)), size=(h, w), **self._up_kwargs)
        x2_4 = ops.interpolate(self.conv2_3(self.pool3(x2)), size=(h, w), **self._up_kwargs)
        x2_5 = ops.interpolate(self.conv2_4(self.pool4(x2)), size=(h, w), **self._up_kwargs)

        # 融合
        x1_fused = self.conv2_5(ops.relu(x2_1 + x2_2 + x2_3))
        x2_fused = self.conv2_6(ops.relu(x2_4 + x2_5))

        out = self.conv3(ops.concat((x1_fused, x2_fused), axis=1))
        return ops.relu(x + out)