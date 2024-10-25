from numbers import Integral

from mindspore import nn, ops


__all__ = ['MobileNet']


class ConvBNLayer(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 num_groups=1,
                 act='relu',
                 conv_lr=1.,
                 conv_decay=0.,
                 norm_decay=0.,
                 norm_type='bn',
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.act = act
        self._conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding,
            group=num_groups,
            has_bias=False)

        if norm_type in ['sync_bn', 'bn']:
            self._batch_norm = nn.BatchNorm2d(out_channels)

    def construct(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        if self.act == "relu":
            x = ops.relu(x)
        elif self.act == "relu6":
            x = ops.relu6(x)
        return x


class DepthwiseSeparable(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels1,
                 out_channels2,
                 num_groups,
                 stride,
                 scale,
                 conv_lr=1.,
                 conv_decay=0.,
                 norm_decay=0.,
                 norm_type='bn',
                 name=None):
        super(DepthwiseSeparable, self).__init__()

        self._depthwise_conv = ConvBNLayer(
            in_channels,
            int(out_channels1 * scale),
            kernel_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            conv_lr=conv_lr,
            conv_decay=conv_decay,
            norm_decay=norm_decay,
            norm_type=norm_type,
            name=name + "_dw")

        self._pointwise_conv = ConvBNLayer(
            int(out_channels1 * scale),
            int(out_channels2 * scale),
            kernel_size=1,
            stride=1,
            padding=0,
            conv_lr=conv_lr,
            conv_decay=conv_decay,
            norm_decay=norm_decay,
            norm_type=norm_type,
            name=name + "_sep")

    def construct(self, x):
        x = self._depthwise_conv(x)
        x = self._pointwise_conv(x)
        return x


class ExtraBlock(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels1,
                 out_channels2,
                 num_groups=1,
                 stride=2,
                 conv_lr=1.,
                 conv_decay=0.,
                 norm_decay=0.,
                 norm_type='bn',
                 name=None):
        super(ExtraBlock, self).__init__()

        self.pointwise_conv = ConvBNLayer(
            in_channels,
            int(out_channels1),
            kernel_size=1,
            stride=1,
            padding=0,
            num_groups=int(num_groups),
            act='relu6',
            conv_lr=conv_lr,
            conv_decay=conv_decay,
            norm_decay=norm_decay,
            norm_type=norm_type,
            name=name + "_extra1")

        self.normal_conv = ConvBNLayer(
            int(out_channels1),
            int(out_channels2),
            kernel_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups),
            act='relu6',
            conv_lr=conv_lr,
            conv_decay=conv_decay,
            norm_decay=norm_decay,
            norm_type=norm_type,
            name=name + "_extra2")

    def construct(self, x):
        x = self.pointwise_conv(x)
        x = self.normal_conv(x)
        return x


class MobileNet(nn.Cell):
    __shared__ = ['norm_type']

    def __init__(self,
                 norm_type='bn',
                 norm_decay=0.,
                 conv_decay=0.,
                 scale=1,
                 conv_learning_rate=1.0,
                 feature_maps=[4, 6, 13],
                 with_extra_blocks=False,
                 extra_block_filters=[[256, 512], [128, 256], [128, 256],
                                      [64, 128]]):
        super(MobileNet, self).__init__()
        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]
        self.feature_maps = feature_maps
        self.with_extra_blocks = with_extra_blocks
        self.extra_block_filters = extra_block_filters

        self._out_channels = []

        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=int(32 * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            conv_lr=conv_learning_rate,
            conv_decay=conv_decay,
            norm_decay=norm_decay,
            norm_type=norm_type,
            name="conv1")

        self.dwsl = []
        self.conv2_1 = DepthwiseSeparable(
                in_channels=int(32 * scale),
                out_channels1=32,
                out_channels2=64,
                num_groups=32,
                stride=1,
                scale=scale,
                conv_lr=conv_learning_rate,
                conv_decay=conv_decay,
                norm_decay=norm_decay,
                norm_type=norm_type,
                name="conv2_1")
        self.dwsl.append(self.conv2_1)
        self._update_out_channels(int(64 * scale), len(self.dwsl), feature_maps)
        self.conv2_2 = DepthwiseSeparable(
                in_channels=int(64 * scale),
                out_channels1=64,
                out_channels2=128,
                num_groups=64,
                stride=2,
                scale=scale,
                conv_lr=conv_learning_rate,
                conv_decay=conv_decay,
                norm_decay=norm_decay,
                norm_type=norm_type,
                name="conv2_2")
        self.dwsl.append(self.conv2_2)
        self._update_out_channels(int(128 * scale), len(self.dwsl), feature_maps)
        # 1/4
        self.conv3_1 = DepthwiseSeparable(
                in_channels=int(128 * scale),
                out_channels1=128,
                out_channels2=128,
                num_groups=128,
                stride=1,
                scale=scale,
                conv_lr=conv_learning_rate,
                conv_decay=conv_decay,
                norm_decay=norm_decay,
                norm_type=norm_type,
                name="conv3_1")
        self.dwsl.append(self.conv3_1)
        self._update_out_channels(int(128 * scale), len(self.dwsl), feature_maps)
        self.conv3_2 = DepthwiseSeparable(
                in_channels=int(128 * scale),
                out_channels1=128,
                out_channels2=256,
                num_groups=128,
                stride=2,
                scale=scale,
                conv_lr=conv_learning_rate,
                conv_decay=conv_decay,
                norm_decay=norm_decay,
                norm_type=norm_type,
                name="conv3_2")
        self.dwsl.append(self.conv3_2)
        self._update_out_channels(int(256 * scale), len(self.dwsl), feature_maps)
        # 1/8
        self.conv4_1 = DepthwiseSeparable(
                in_channels=int(256 * scale),
                out_channels1=256,
                out_channels2=256,
                num_groups=256,
                stride=1,
                scale=scale,
                conv_lr=conv_learning_rate,
                conv_decay=conv_decay,
                norm_decay=norm_decay,
                norm_type=norm_type,
                name="conv4_1")
        self.dwsl.append(self.conv4_1)
        self._update_out_channels(int(256 * scale), len(self.dwsl), feature_maps)
        self.conv4_2 = DepthwiseSeparable(
                in_channels=int(256 * scale),
                out_channels1=256,
                out_channels2=512,
                num_groups=256,
                stride=2,
                scale=scale,
                conv_lr=conv_learning_rate,
                conv_decay=conv_decay,
                norm_decay=norm_decay,
                norm_type=norm_type,
                name="conv4_2")
        self.dwsl.append(self.conv4_2)
        self._update_out_channels(int(512 * scale), len(self.dwsl), feature_maps)
        # 1/16
        for i in range(5):
            self.insert_child_to_cell("conv5_" + str(i + 1), DepthwiseSeparable(
                                                            in_channels=int(512 * scale),
                                                            out_channels1=512,
                                                            out_channels2=512,
                                                            num_groups=512,
                                                            stride=1,
                                                            scale=scale,
                                                            conv_lr=conv_learning_rate,
                                                            conv_decay=conv_decay,
                                                            norm_decay=norm_decay,
                                                            norm_type=norm_type,
                                                            name="conv5_" + str(i + 1)))
            self.dwsl.append(getattr(self, "conv5_" + str(i + 1)))
            self._update_out_channels(int(512 * scale), len(self.dwsl), feature_maps)
        self.conv5_6 = DepthwiseSeparable(
                in_channels=int(512 * scale),
                out_channels1=512,
                out_channels2=1024,
                num_groups=512,
                stride=2,
                scale=scale,
                conv_lr=conv_learning_rate,
                conv_decay=conv_decay,
                norm_decay=norm_decay,
                norm_type=norm_type,
                name="conv5_6")
        self.dwsl.append(self.conv5_6)
        self._update_out_channels(int(1024 * scale), len(self.dwsl), feature_maps)
        # 1/32
        self.conv6 = DepthwiseSeparable(
                in_channels=int(1024 * scale),
                out_channels1=1024,
                out_channels2=1024,
                num_groups=1024,
                stride=1,
                scale=scale,
                conv_lr=conv_learning_rate,
                conv_decay=conv_decay,
                norm_decay=norm_decay,
                norm_type=norm_type,
                name="conv6")
        self.dwsl.append(self.conv6)
        self._update_out_channels(int(1024 * scale), len(self.dwsl), feature_maps)

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self._out_channels.append(channel)

    def construct(self, inputs):
        outs = ()
        y = self.conv1(inputs)
        for i, block in enumerate(self.dwsl):
            y = block(y)
            if i + 1 in self.feature_maps:
                outs += (y,)

        if not self.with_extra_blocks:
            return outs

        y = outs[-1]
        for i, block in enumerate(self.extra_blocks):
            idx = i + len(self.dwsl)
            y = block(y)
            if idx + 1 in self.feature_maps:
                outs += (y,)
        return outs
