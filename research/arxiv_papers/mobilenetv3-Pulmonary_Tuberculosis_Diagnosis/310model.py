import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

class HSigmoid(nn.Cell):
    def __init__(self):
        super(HSigmoid, self).__init__()
        self.relu6 = nn.ReLU6()

    def construct(self, x):
        return self.relu6(x + 3.) / 6.

class HSwish(nn.Cell):
    def __init__(self):
        super(HSwish, self).__init__()
        self.hsigmoid = HSigmoid()

    def construct(self, x):
        return x * self.hsigmoid(x)

class SEBlock(nn.Cell):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.fc1 = nn.Dense(in_channels, in_channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        out = self.avg_pool(x, (2, 3))
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.shape[0], out.shape[1], 1, 1)
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
        self.hs = HSwish()  # 使用自定义HSwish
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=False) if in_channels != out_channels else None
        self.bn_identity = nn.BatchNorm2d(out_channels) if in_channels != out_channels else None

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.hs(out) if self.use_hs else self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.hs(out) if self.use_hs else self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)
        identity = self.identity(x) if self.identity else x
        identity = self.bn_identity(identity) if self.bn_identity else identity
        out = out + identity
        return self.hs(out) if self.use_hs else self.relu(out)

class MobileNetV3(nn.Cell):
    def __init__(self, num_classes=2, mode='large', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        self.cfgs = [
            # 修改所有HS层使用RE模式（避免预训练参数依赖）
            [3, 16, 16, False, 'RE', 1],
            [3, 64, 24, False, 'RE', 2],
            [3, 72, 24, False, 'RE', 1],
            [5, 72, 40, True, 'RE', 2],
            [5, 120, 40, True, 'RE', 1],
            [5, 120, 40, True, 'RE', 1],
            [3, 240, 80,  False, 'HS', 2],
            [3, 200, 80,  False, 'HS', 1],
            [3, 184, 80,  False, 'HS', 1],
            [3, 184, 80,  False, 'HS', 1],
            [3, 480, 112, True,  'HS', 1],
            [3, 672, 112, True,  'HS', 1],
            [5, 672, 160, True,  'HS', 2],
            [5, 960, 160, True,  'HS', 1],
            [5, 960, 160, True,  'HS', 1],
        ] if mode == 'large' else [
            [3, 16, 16, True, 'RE', 2],
            [3, 72, 24, False, 'RE', 2],
            [3, 88, 24, False, 'RE', 1],
            [5, 96,  40,  True,  'HS', 2],
            [5, 240, 40,  True,  'HS', 1],
            [5, 240, 40,  True,  'HS', 1],
            [5, 120, 48,  True,  'HS', 1],
            [5, 144, 48,  True,  'HS', 1],
            [5, 288, 96,  True,  'HS', 2],
            [5, 576, 96,  True,  'HS', 1],
            [5, 576, 96,  True,  'HS', 1],
        ]
        self.last_channels = 960 if mode == 'large' else 576
        input_channel = self._make_divisible(16 * width_mult, 8)
        layers = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU()
        ]
        for k, exp, c, se, nl, s in self.cfgs:
            output_channel = self._make_divisible(c * width_mult, 8)
            exp_channel = self._make_divisible(exp * width_mult, 8)
            layers.append(Bottleneck(
                input_channel, output_channel, exp_channel, k, s, se, nl == 'HS'
            ))
            input_channel = output_channel
        last_channel = self._make_divisible(self.last_channels * width_mult, 8) if width_mult > 1.0 else self.last_channels
        layers += [
            nn.Conv2d(input_channel, last_channel, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU()
        ]
        self.features = nn.SequentialCell(layers)
        self.avgpool = ops.ReduceMean(keep_dims=True)
        self.classifier = nn.Dense(last_channel, num_classes)

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def construct(self, x):
        x = self.features(x)
        x = self.avgpool(x, (2, 3))
        x = x.view(x.shape[0], -1)
        return self.classifier(x)