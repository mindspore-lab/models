import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore import set_context

set_context(mode=ms.GRAPH_MODE, device_target="Ascend", enable_graph_kernel=True)

class SEBlock(nn.Cell):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.fc1 = nn.Dense(in_channels, in_channels // reduction, has_bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(in_channels // reduction, in_channels, has_bias=True)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        out = self.avg_pool(x, (2, 3))  # 全局平均池化，将空间维度压缩为 1x1
        out = out.view(out.shape[0], -1)  # 展平
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.shape[0], out.shape[1], 1, 1)  # 恢复形状
        return x * out


class Bottleneck(nn.Cell):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, stride, use_se, use_hs):
        super(Bottleneck, self).__init__()
        self.use_se = use_se
        self.use_hs = use_hs
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, exp_size, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(exp_size)
        self.conv2 = nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride,
                           pad_mode='same', group=exp_size, has_bias=False)
        self.bn2 = nn.BatchNorm2d(exp_size)
        self.conv3 = nn.Conv2d(exp_size, out_channels, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else None
        self.relu = nn.ReLU()
        self.hs = nn.HSwish()
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=False) if in_channels!= out_channels else None
        self.bn_identity = nn.BatchNorm2d(out_channels) if in_channels!= out_channels else None

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        if self.use_hs:
            out = self.hs(out)
        else:
            out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_hs:
            out = self.hs(out)
        else:
            out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)
        if self.identity is not None:
            identity = self.identity(x)
            identity = self.bn_identity(identity)
        else:
            identity = x
        out = out + identity  # 残差连接
        if self.use_hs:
            out = self.hs(out)
        else:
            out = self.relu(out)
        return out


class MobileNetV3(nn.Cell):
    def __init__(self, num_classes=2, mode='large', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        self.cfgs = []
        if mode == 'large':
            self.cfgs = [
                # k, exp, c,  se,     nl,  s
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1],
            ]
            self.last_channels = 960
        else:  # mode == 'small'
            self.cfgs = [
                # k, exp, c,  se,     nl,  s
                [3, 16, 16, True, 'RE', 2],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1],
            ]
            self.last_channels = 576
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = []
        layers.append(nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, pad_mode='pad', has_bias=False))
        layers.append(nn.BatchNorm2d(input_channel))
        if mode == 'large':
            layers.append(nn.HSwish())
        else:
            layers.append(nn.ReLU())
        for k, exp, c, se, nl, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_channel = _make_divisible(exp * width_mult, 8)
            use_hs = nl == 'HS'
            use_se = se
            layers.append(Bottleneck(input_channel, output_channel, exp_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        last_channel = _make_divisible(self.last_channels * width_mult, 8) if width_mult > 1.0 else self.last_channels
        layers.append(nn.Conv2d(input_channel, last_channel, kernel_size=1, has_bias=False))
        layers.append(nn.BatchNorm2d(last_channel))
        if mode == 'large':
            layers.append(nn.HSwish())
        else:
            layers.append(nn.ReLU())
        self.features = nn.SequentialCell(layers)
        self.avgpool = ops.ReduceMean(keep_dims=True)
        self.classifier = nn.Dense(last_channel, num_classes)

    def construct(self, x):
        x = self.features(x)
        x = self.avgpool(x, (2, 3))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v