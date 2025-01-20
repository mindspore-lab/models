from mindspore import nn

class ConvNormLayer(nn.Cell):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding=0, pad_mode='same', act=False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode=pad_mode,
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.act = act

    def construct(self, x):
        y = self.conv(x)
        y = self.norm(y)
        if self.act:
            y = self.relu(y)
        return y


class DeConvNormLayer(nn.Cell):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding=0, pad_mode='same', act=False
    ):
        super().__init__()
        self.deconv = nn.Conv2dTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode=pad_mode,
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.act = act

    def construct(self, x):
        y = self.deconv(x)
        y = self.norm(y)
        if self.act:
            y = self.relu(y)
        return y