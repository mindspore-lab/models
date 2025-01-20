from mindspore import nn

from ._registry import register_backbone, register_backbone_class
from models.networks.resnet_cells import ConvNormLayer

__all__ = ['E2ePgResNet', 'pgnet_backbone']


class Bottleneck(nn.Cell):
    """
    A wrapper of the original PGNet  described in
    `"PGNet: Real-time Arbitrarily-Shaped Text Spotting with Point Gathering NetWork" <https://arxiv.org/abs/1905.02244>`_ 
    that extracts features maps from differentstages.

    Examples:
        Initializing PGNet for feature extraction:
        >>> model = E2ePgResNet(Bottleneck, [3, 4, 6, 3, 3])
    """
    expansion = 4   
    def __init__(self, in_channel, out_channel, stride=1, shortcut=True):
        super().__init__()
        self.conv0 = ConvNormLayer(in_channels=in_channel, out_channels=out_channel, 
                                   kernel_size=1, stride=1, act=True)
        self.conv1 = ConvNormLayer(in_channels=out_channel, out_channels=out_channel, 
                                   kernel_size=3, stride=stride, padding=1, pad_mode='pad', act=True)
        self.conv2 = ConvNormLayer(in_channels=out_channel, out_channels=out_channel*self.expansion, 
                                   kernel_size=1, stride=1)
        self.relu = nn.ReLU()

        self.shortcut = shortcut
        if not shortcut:
            self.short = ConvNormLayer(
                in_channels=in_channel,
                out_channels=out_channel * self.expansion,
                kernel_size=1,
                stride=stride)

    def construct(self, x):
        identity = x
        if not self.shortcut:
            identity = self.short(x)

        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out
    
@register_backbone_class
class E2ePgResNet(nn.Cell):
    def __init__(self, block, block_num, in_channels=3):
        super(E2ePgResNet, self).__init__()
        self.in_channels = (64, 256, 512, 1024, 2048)
        self.block_num = block_num
        self.out_channels = [3, 64]

        self.conv1_1 = ConvNormLayer(in_channels, out_channels=64, kernel_size=7, 
                                     stride=2, padding=3, pad_mode='pad', act=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, pad_mode='pad')
        self.layer1 = self._make_layer(block=block, channel=64, block_idx=0,
                                       stride=1)
        self.layer2 = self._make_layer(block=block, channel=128, block_idx=1,
                                       stride=2)
        self.layer3 = self._make_layer(block=block, channel=256, block_idx=2,
                                       stride=2)
        self.layer4 = self._make_layer(block=block, channel=512, block_idx=3,
                                       stride=2)
        self.layer5 = self._make_layer(block=block, channel=512, block_idx=4,
                                       stride=2)

    def _make_layer(self, block, channel, block_idx, stride):
        layers = []
        layers.append(block(in_channel=self.in_channels[block_idx], out_channel=channel, 
                            shortcut=False, stride=stride))
        input_channels = channel * block.expansion
        
        for _ in range(1, self.block_num[block_idx]):
            layers.append(block(input_channels, out_channel=channel, stride=1))
        
        self.out_channels.append(input_channels)

        return nn.SequentialCell(layers)

    def construct(self, x):
        x0 = self.conv1_1(x)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return [x, x0, x1, x2, x3, x4, x5]

@register_backbone
def pgnet_backbone(pretrained: bool = False, **kwargs) -> E2ePgResNet:
    if pretrained is True:
        raise NotImplementedError(
            "The default pretrained checkpoint for `e2e_pgresnet` backbone does not exist."
        )

    model = E2ePgResNet(block=Bottleneck, block_num=[3, 4, 6, 3, 3], **kwargs)
    return model
