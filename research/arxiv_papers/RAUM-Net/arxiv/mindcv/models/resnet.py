"""
MindSpore implementation of `ResNet` with Mamba integration.
Refer to Deep Residual Learning for Image Recognition.
"""

from typing import List, Optional, Type, Union

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .helpers import build_model_with_cfg
from .layers.pooling import GlobalAvgPooling
from .registry import register_model

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x4d",
    "resnext101_64x4d",
    "resnext152_64x4d",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv1",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "resnet18": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet18-1e65cd21.ckpt"),
    "resnet34": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet34-f297d27e.ckpt"),
    "resnet50": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet50-e0733ab8.ckpt"),
    "resnet101": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet101-689c5e77.ckpt"),
    "resnet152": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet152-beb689d8.ckpt"),
    "resnext50_32x4d": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnext/resnext50_32x4d-af8aba16.ckpt"),
    "resnext101_32x4d": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/resnext/resnext101_32x4d-3c1e9c51.ckpt"
    ),
    "resnext101_64x4d": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/resnext/resnext101_64x4d-8929255b.ckpt"
    ),
    "resnext152_64x4d": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/resnext/resnext152_64x4d-3aba275c.ckpt"
    ),
}


class BasicBlock(nn.Cell):
    """Basic block of ResNet"""
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
        use_mamba: bool = False,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        assert groups == 1, "BasicBlock only supports groups=1"
        assert base_width == 64, "BasicBlock only supports base_width=64"

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=stride, padding=1, pad_mode="pad")
        self.bn1 = norm(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, pad_mode="pad")
        self.bn2 = norm(channels)
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    """Bottleneck with Mamba integration"""
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
        use_mamba: bool = False,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        width = int(channels * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1)
        self.bn1 = norm(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, pad_mode="pad", group=groups)
        self.bn2 = norm(width)
        self.conv3 = nn.Conv2d(width, channels * self.expansion,
                               kernel_size=1, stride=1)
        self.bn3 = norm(channels * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample = down_sample
        
        self.use_mamba = use_mamba
        if use_mamba:
            self.mamba = MambaBlock(dim=channels * self.expansion)

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.use_mamba:
            out = self.mamba(out)
        
        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class MambaBlock(nn.Cell):
    """Streamlined and efficient Mamba block: focuses on spatial relationship modeling and training stability"""
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        
        self.d_inner = max(dim // 2, 64)
        
        self.in_proj = nn.Conv2d(dim, self.d_inner, kernel_size=1, 
                                 weight_init=init.HeNormal())
        self.in_norm = nn.BatchNorm2d(self.d_inner, gamma_init='zeros')
        
        self.spatial = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=5,
                                stride=1, padding=2, pad_mode="pad")
        self.spatial_norm = nn.BatchNorm2d(self.d_inner)
        
        self.out_proj = nn.Conv2d(self.d_inner, dim, kernel_size=1)
        self.out_norm = nn.BatchNorm2d(dim, gamma_init='zeros')
        
        self.relu = nn.ReLU()
        
        self.scale = 0.1
    
    def construct(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.in_proj(x)
        out = self.in_norm(out)
        out = self.relu(out)
        
        out = self.spatial(out)
        out = self.spatial_norm(out)
        out = self.relu(out)
        
        out = self.out_proj(out)
        out = self.out_norm(out)
        
        out = identity + self.scale * out
        out = self.relu(out)
        
        return out


class MambaBottleneck(nn.Cell):
    """Adaptive Mamba Bottleneck: dynamically adjusts Mamba influence based on network position"""
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
        use_mamba: bool = True,
        layer_index: int = 0,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        width = int(channels * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1)
        self.bn1 = norm(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, pad_mode="pad", group=groups)
        self.bn2 = norm(width)
        self.conv3 = nn.Conv2d(width, channels * self.expansion,
                               kernel_size=1, stride=1)
        self.bn3 = norm(channels * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample = down_sample
        
        self.use_mamba = use_mamba
        if use_mamba:
            self.mamba = MambaBlock(dim=channels * self.expansion)

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.use_mamba:
            out = self.mamba(out)
        
        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """Optimized ResNet-Mamba integration architecture"""
    
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        use_mamba: bool = True,
        cifar_mode: bool = False,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        
        self.use_mamba_in_layer = [False, False, True, True] if use_mamba else [False, False, False, False]
        
        self.norm = norm
        self.groups = groups
        self.base_width = base_width
        self.in_channels = 64
        
        if cifar_mode:
            self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=3, stride=1, 
                                  padding=1, pad_mode="pad")
            self.max_pool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=7, stride=2, 
                                  padding=3, pad_mode="pad")
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
            
        self.bn1 = norm(self.in_channels)
        self.relu = nn.ReLU()
        
        self.layer1 = self._make_layer(block, 64, layers[0], use_mamba=self.use_mamba_in_layer[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_mamba=self.use_mamba_in_layer[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_mamba=self.use_mamba_in_layer[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_mamba=self.use_mamba_in_layer[3])
        
        self.pool = GlobalAvgPooling()
        self.num_features = 512 * block.expansion
        self.classifier = nn.Dense(self.num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Constant(0), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer(init.Constant(1), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer(init.Constant(0), cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(), cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Constant(0), cell.bias.shape, cell.bias.dtype))

    def _make_layer(self, block, channels, block_nums, stride=1, use_mamba=False):
        """Build ResNet layer

        Args:
            block (Cell): block type (BasicBlock or Bottleneck)
            channels (int): output channels
            block_nums (int): number of blocks
            stride (int): stride of the first block
            use_mamba (bool): whether to use Mamba

        Returns:
            SequentialCell: built layer structure
        """
        down_sample = None

        if stride != 1 or self.in_channels != channels * block.expansion:
            down_sample = nn.SequentialCell([
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride),
                self.norm(channels * block.expansion)
            ])
        
        layers = []
        if block is Bottleneck:
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    stride=stride,
                    down_sample=down_sample,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm=self.norm,
                    use_mamba=use_mamba
                )
            )
        else:
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    stride=stride,
                    down_sample=down_sample,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm=self.norm
                )
            )
        
        self.in_channels = channels * block.expansion
        
        for _ in range(1, block_nums):
            if block is Bottleneck:
                layers.append(
                    block(
                        self.in_channels,
                        channels,
                        groups=self.groups,
                        base_width=self.base_width,
                        norm=self.norm,
                        use_mamba=use_mamba
                    )
                )
            else:
                layers.append(
                    block(
                        self.in_channels, 
                        channels,
                        groups=self.groups,
                        base_width=self.base_width,
                        norm=self.norm
                    )
                )
        
        return nn.SequentialCell(layers)

    def forward_features(self, x: Tensor) -> Tensor:
        """Network forward feature extraction."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_resnet(pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, pretrained, **kwargs)


@register_model
def resnet18(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 18 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnet18"]
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels,
                      **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnet34(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 34 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnet34"]
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels,
                      **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnet50(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    cifar_mode: bool = False,
    use_mamba: bool = True,
    **kwargs
) -> ResNet:
    """Get 50 layers ResNet model.
    
    Args:
        pretrained: Whether to download and load the pre-trained model. Default: False.
        num_classes: The number of classification. Default: 1000.
        in_channels: The input channels. Default: 3.
        cifar_mode: Whether to use CIFAR optimized architecture. Default: False.
        use_mamba: Whether to use Mamba blocks. Default: True.
        
    Returns:
        ResNet network.
    """
    default_cfg = default_cfgs["resnet50"]
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels,
        cifar_mode=cifar_mode,
        use_mamba=use_mamba,
        **kwargs
    )
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnet101(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 101 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnet101"]
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels,
                      **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnet152(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 152 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnet152"]
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], num_classes=num_classes, in_channels=in_channels,
                      **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnext50_32x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 50 layers ResNeXt model with 32 groups of GPConv.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnext50_32x4d"]
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], groups=32, base_width=4, num_classes=num_classes,
                      in_channels=in_channels, **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnext101_32x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 101 layers ResNeXt model with 32 groups of GPConv.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnext101_32x4d"]
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, base_width=4, num_classes=num_classes,
                      in_channels=in_channels, **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnext101_64x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 101 layers ResNeXt model with 64 groups of GPConv.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnext101_64x4d"]
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=64, base_width=4, num_classes=num_classes,
                      in_channels=in_channels, **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnext152_64x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["resnext152_64x4d"]
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], groups=64, base_width=4, num_classes=num_classes,
                      in_channels=in_channels, **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))
