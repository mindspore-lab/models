from typing import Optional, Type, Union, List

import mindspore.nn as nn
from mindspore.ops import operations as P
# from mindvision.classification.models.classifiers import BaseClassifier
# from mindvision.classification.models.head import DenseHead
# from mindvision.classification.models.neck import GlobalAvgPooling
# from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from load_pretrained_model import LoadPretrainedModel


model_urls = {
    # lenet series
    "lenet": "https://download.mindspore.cn/vision/classification/lenet_mnist.ckpt",
    # resnet series
    "resnet18": "https://download.mindspore.cn/vision/classification/resnet18_224.ckpt",
    "resnet34": "https://download.mindspore.cn/vision/classification/resnet34_224.ckpt",
    "resnet50": "https://download.mindspore.cn/vision/classification/resnet50_224.ckpt",
    "resnet101": "https://download.mindspore.cn/vision/classification/resnet101_224.ckpt",
    "resnet152": "https://download.mindspore.cn/vision/classification/resnet152_224.ckpt",
    # mobilenet_v2 series
    "mobilenet_v2_1.4_224": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.4_224.ckpt",
    "mobilenet_v2_1.3_224": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.3_224.ckpt",
    "mobilenet_v2_1.0_224": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_224.ckpt",
    "mobilenet_v2_1.0_192": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_192.ckpt",
    "mobilenet_v2_1.0_160": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_160.ckpt",
    "mobilenet_v2_1.0_128": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_128.ckpt",
    "mobilenet_v2_1.0_96": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_96.ckpt",
    "mobilenet_v2_0.75_224": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.75_224.ckpt",
    "mobilenet_v2_0.75_192": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.75_192.ckpt",
    "mobilenet_v2_0.75_160": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.75_160.ckpt",
    "mobilenet_v2_0.75_128": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.75_128.ckpt",
    "mobilenet_v2_0.75_96": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.75_96.ckpt",
    "mobilenet_v2_0.5_224": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.5_224.ckpt",
    "mobilenet_v2_0.5_192": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.5_192.ckpt",
    "mobilenet_v2_0.5_160": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.5_160.ckpt",
    "mobilenet_v2_0.5_128": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.5_128.ckpt",
    "mobilenet_v2_0.5_96": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.5_96.ckpt",
    "mobilenet_v2_0.35_224": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.35_224.ckpt",
    "mobilenet_v2_0.35_192": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.35_192.ckpt",
    "mobilenet_v2_0.35_160": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.35_160.ckpt",
    "mobilenet_v2_0.35_128": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.35_128.ckpt",
    "mobilenet_v2_0.35_96": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.35_96.ckpt",
    # ViT series
    "vit_b_16_224": "https://download.mindspore.cn/vision/classification/vit_b_16_224.ckpt",
    "vit_b_16_384": "https://download.mindspore.cn/vision/classification/vit_b_16_384.ckpt",
    "vit_l_16_224": "https://download.mindspore.cn/vision/classification/vit_l_16_224.ckpt",
    "vit_l_16_384": "https://download.mindspore.cn/vision/classification/vit_l_16_384.ckpt",
    "vit_b_32_224": "https://download.mindspore.cn/vision/classification/vit_b_32_224_tv.ckpt",
    "vit_b_32_384": "https://download.mindspore.cn/vision/classification/vit_b_32_384.ckpt",
    "vit_l_32_224": "https://download.mindspore.cn/vision/classification/vit_l_32_224_tv.ckpt",
    # EfficientNet series
    # eg. b0(Model name)_1(Width Coefficient)_1(Depth Coefficient)_224(Resolution)_0.2(Dropout Rate).ckpt
    "efficientnet_b0": "https://download.mindspore.cn/vision/classification/efficientnet_b0_1_1_224_0.2.ckpt",
    "efficientnet_b1": "https://download.mindspore.cn/vision/classification/efficientnet_b1_1_1.1_240_0.2.ckpt",
    "efficientnet_b2": "https://download.mindspore.cn/vision/classification/efficientnet_b2_1.1_1.2_260_0.3.ckpt",
    "efficientnet_b3": "https://download.mindspore.cn/vision/classification/efficientnet_b3_1.2_1.4_300_0.3.ckpt",
    "efficientnet_b4": "https://download.mindspore.cn/vision/classification/efficientnet_b4_1.4_1.8_380_0.4.ckpt",
    "efficientnet_b5": "https://download.mindspore.cn/vision/classification/efficientnet_b5_1.6_2.2_456_0.4.ckpt",
    "efficientnet_b6": "https://download.mindspore.cn/vision/classification/efficientnet_b6_1.8_2.6_528_0.5.ckpt",
    "efficientnet_b7": "https://download.mindspore.cn/vision/classification/efficientnet_b7_2.0_3.1_600_0.5.ckpt",
}


class ResidualBlockBase(nn.Cell):
    """
    ResNet residual block base definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        group (int): Group convolutions. Default: 1.
        base_width (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
        down_sample (nn.Cell, optional): Downsample structure. Default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlockBase(3, 256, stride=2)
    """

    expansion: int = 1

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 stride: int = 1,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None
                 ) -> None:
        super(ResidualBlockBase, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d
        assert group != 1 or base_width == 64, "ResidualBlockBase only supports groups=1 and base_width=64"
        self.conv1 = ConvNormActivation(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            norm=norm)
        self.conv2 = ConvNormActivation(
            out_channel,
            out_channel,
            kernel_size=3,
            norm=norm,
            activation=None)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        """ResidualBlockBase construct."""
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.down_sample:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)

        return out



class ConvNormActivation(nn.Cell):
    """
    Convolution/Depthwise fused with normalization and activation blocks definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution
        layer. Default: nn.BatchNorm2d.
        activation (nn.Cell, optional): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.ReLU.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> conv = ConvNormActivation(16, 256, kernel_size=1, stride=1, groups=1)
    """

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm: Optional[nn.Cell] = nn.BatchNorm2d,
                 activation: Optional[nn.Cell] = nn.ReLU,
                 has_bias: bool = False
                 ) -> None:
        super(ConvNormActivation, self).__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                pad_mode='pad',
                padding=padding,
                group=groups,
                has_bias=has_bias
            )
        ]

        if norm:
            layers.append(norm(out_planes))
        if activation:
            layers.append(activation())

        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output
    

class ResidualBlock(nn.Cell):
    """
    ResNet residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the second convolutional layer. Default: 1.
        group (int): Group convolutions. Default: 1.
        base_width (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
        down_sample (nn.Cell, optional): Downsample structure. Default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> from mindvision.classification.models.backbones import ResidualBlock
        >>> ResidualBlock(3, 256, stride=2)
    """

    expansion: int = 4

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 stride: int = 1,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None
                 ) -> None:
        super(ResidualBlock, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d
        out_channel = int(out_channel * (base_width / 64.0)) * group

        self.conv1 = ConvNormActivation(
            in_channel, out_channel, kernel_size=1, norm=norm)
        self.conv2 = ConvNormActivation(
            out_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            groups=group,
            norm=norm)
        self.conv3 = ConvNormActivation(
            out_channel,
            out_channel *
            self.expansion,
            kernel_size=1,
            norm=norm,
            activation=None)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        """ResidualBlock construct."""
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.down_sample:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Type[Union[ResidualBlockBase, ResidualBlock]]): THe block for network.
        layer_nums (list): The numbers of block in different layers.
        group (int): The number of Group convolutions. Default: 1.
        base_width (int): The width of per group. Default: 64.
        norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, 2048, 7, 7)`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindvision.classification.models.backbones import ResNet, ResidualBlock
        >>> net = ResNet(ResidualBlock, [3, 4, 23, 3])
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 2048, 7, 7)

    About ResNet:

    The ResNet is to ease the training of networks that are substantially deeper than those used previously.
    The model explicitly reformulate the layers as learning residual functions with reference to the layer inputs,
    instead of learning unreferenced functions.

    Citation:

    .. code-block::

        @article{2016Deep,
        title={Deep Residual Learning for Image Recognition},
        author={ He, K.  and  Zhang, X.  and  Ren, S.  and  Sun, J. },
        journal={IEEE},
        year={2016},
        }
    """

    def __init__(self,
                 block: Type[Union[ResidualBlockBase, ResidualBlock]],
                 layer_nums: List[int],
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None
                 ) -> None:
        super(ResNet, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d
        self.norm = norm
        self.in_channels = 64
        self.group = group
        self.base_with = base_width
        self.conv1 = ConvNormActivation(
            3, self.in_channels, kernel_size=7, stride=2, norm=norm)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layer_nums[0])
        self.layer2 = self._make_layer(block, 128, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_nums[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_nums[3], stride=2)

    def _make_layer(self,
                    block: Type[Union[ResidualBlockBase, ResidualBlock]],
                    channel: int,
                    block_nums: int,
                    stride: int = 1
                    ):
        """Block layers."""
        down_sample = None

        if stride != 1 or self.in_channels != self.in_channels * block.expansion:
            down_sample = ConvNormActivation(
                self.in_channels,
                channel * block.expansion,
                kernel_size=1,
                stride=stride,
                norm=self.norm,
                activation=None)
        layers = []
        layers.append(
            block(
                self.in_channels,
                channel,
                stride=stride,
                down_sample=down_sample,
                group=self.group,
                base_width=self.base_with,
                norm=self.norm
            )
        )
        self.in_channels = channel * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.in_channels,
                    channel,
                    group=self.group,
                    base_width=self.base_with,
                    norm=self.norm
                )
            )

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class GlobalAvgPooling(nn.Cell):
    """
    Global avg pooling definition.

    Args:

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GlobalAvgPooling()
    """

    def __init__(self,
                 keep_dims: bool = False
                 ) -> None:
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(keep_dims=keep_dims)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class BaseClassifier(nn.Cell):
    """
    generate classifier
    """

    def __init__(self, backbone, neck=None, head=None):
        super(BaseClassifier, self).__init__()
        self.backbone =  backbone
        if neck:
            self.neck = neck
            self.with_neck = True
        else:
            self.with_neck = False
        if head:
            self.head = head
            self.with_head = True
        else:
            self.with_head = False

    def construct(self, x):
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        if self.with_head:
            x = self.head(x)
        return x



def _resnet(arch: str,
            block: Type[Union[ResidualBlockBase, ResidualBlock]],
            layers: List[int],
            num_classes: int,
            pretrained: bool,
            input_channel: int,
            group: int = 1,
            base_width: int = 64,
            norm: Optional[nn.Cell] = None
            ) -> ResNet:
    """ResNet architecture."""
    backbone = ResNet(
        block=block,
        layer_nums=layers,
        group=group,
        base_width=base_width,
        norm=norm
    )
    neck = GlobalAvgPooling()
    head = DenseHead(input_channel=input_channel, num_classes=num_classes)
    model = BaseClassifier(backbone, neck, head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model



class DenseHead(nn.Cell):
    """
    LinearClsHead architecture.

    Args:
        input_channel (int): The number of input channel.
        num_classes (int): Number of classes.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[str, Cell, Primitive]): activate function applied to the output. Eg. `ReLU`. Default: None.
        keep_prob (float): Dropout keeping rate, between [0, 1]. E.g. rate=0.9, means dropping out 10% of input.
            Default: 1.0.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self,
                 input_channel: int,
                 num_classes: int,
                 has_bias: bool = True,
                 activation: Optional[Union[str, nn.Cell]] = None,
                 keep_prob: float = 1.0
                 ) -> None:
        super(DenseHead, self).__init__()

        self.dropout = nn.Dropout(keep_prob)
        self.dense = nn.Dense(input_channel, num_classes, has_bias=has_bias, activation=activation)

    def construct(self, x):
        if self.training:
            x = self.dropout(x)
        x = self.dense(x)
        return x

def resnet50(num_classes: int = 1000,
             pretrained: bool = False,
             group: int = 1,
             base_width: int = 64,
             norm: Optional[nn.Cell] = None
             ) -> ResNet:
   
    return _resnet(
        "resnet50", ResidualBlock, [
            3, 4, 6, 3], num_classes, pretrained, 2048, group, base_width, norm)



