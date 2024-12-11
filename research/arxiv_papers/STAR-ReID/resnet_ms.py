from typing import Type, List, Optional
from mindspore import nn
from mindspore.common.initializer import Normal
from mindspore import load_checkpoint, load_param_into_net
import mindspore as ms
from mindspore import Tensor
import numpy as np

weight_init = Normal(mean=0, sigma=0.02)
gamma_init = Normal(mean=1, sigma=0.02)

class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, down_sample: Optional[nn.Cell] = None, dilation: int = 1) -> None:
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=1, weight_init=weight_init, has_bias=False, pad_mode='pad', padding=0)
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation,
                               weight_init=weight_init, has_bias=False, pad_mode='pad')
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion,
                               kernel_size=1, weight_init=weight_init, has_bias=False, pad_mode='pad', padding=0)
        self.norm3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out

def make_layer(last_out_channel, block: Type[ResidualBlock],
               channel: int, block_nums: int, stride: int = 1, dilation: int = 1):
    down_sample = None

    if stride != 1 or last_out_channel != channel * block.expansion:
        down_sample = nn.SequentialCell([
            nn.Conv2d(
                last_out_channel,
                channel * block.expansion,
                kernel_size=1,
                stride=stride,
                weight_init=weight_init,
                has_bias=False,
                pad_mode='pad',
                padding=0
            ),
            nn.BatchNorm2d(channel * block.expansion, gamma_init=gamma_init)
        ])

    layers = []
    layers.append(block(
        last_out_channel,
        channel,
        stride=stride,
        down_sample=down_sample,
        dilation=dilation
    ))

    in_channel = channel * block.expansion
    for _ in range(1, block_nums):
        layers.append(block(in_channel, channel, dilation=dilation))

    return nn.SequentialCell(layers)

class ResNet(nn.Cell):
    def __init__(self, block: Type[ResidualBlock],
                 layer_nums: List[int], last_conv_stride: int = 2, last_conv_dilation: int = 1) -> None:
        super(ResNet, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3,
            weight_init=weight_init, has_bias=False, pad_mode='pad'
        )
        self.norm = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.layer1 = make_layer(64, block, 64, layer_nums[0])
        self.layer2 = make_layer(64 * block.expansion, block, 128, layer_nums[1], stride=2)
        self.layer3 = make_layer(128 * block.expansion, block, 256, layer_nums[2], stride=2)
        self.layer4 = make_layer(
            256 * block.expansion,
            block,
            512,
            layer_nums[3],
            stride=last_conv_stride,
            dilation=last_conv_dilation
        )

    def construct(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
def remove_fc(state_dict):
  """Remove the fc layer parameters from state_dict."""
  # for key, value in state_dict.items():
  for key, value in list(state_dict.items()):
    if key.startswith('fc.'):
      del state_dict[key]
  return state_dict

def _resnet(block: Type[ResidualBlock],
            layers: List[int], pretrained: bool, pretrained_ckpt: str,
            last_conv_stride: int, last_conv_dilation: int):
    model = ResNet(
        block=block,
        layer_nums=layers,
        last_conv_stride=last_conv_stride,
        last_conv_dilation=last_conv_dilation
    )

    if pretrained:
        # Load pre-trained model
        param_dict = load_checkpoint(pretrained_ckpt)
        # Remove fully connected layer parameters from pre-trained model
        param_dict = {key: value for key, value in param_dict.items() if not key.startswith('fc.')}
        load_param_into_net(model, remove_fc(param_dict), strict_load=False)

    return model

def resnet50(pretrained: bool = False, last_conv_stride: int = 2, last_conv_dilation: int = 1):
    "ResNet50 model"
    pretrained_ckpt = "./resnet50_224_new.ckpt"  # Update this path to your checkpoint file
    return _resnet(
        block=ResidualBlock,
        layers=[3, 4, 6, 3],
        pretrained=pretrained,
        pretrained_ckpt=pretrained_ckpt,
        last_conv_stride=last_conv_stride,
        last_conv_dilation=last_conv_dilation
    )

if __name__ == '__main__':           
    # Create a random input Tensor with batch size 1, 3 channels, and size 224x224
    x = Tensor(np.random.randn(1, 3, 224, 224), ms.float32)
    # Instantiate the network with the desired parameters
    net = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
    output = net(x)
    print("Output shape:", output.shape)
