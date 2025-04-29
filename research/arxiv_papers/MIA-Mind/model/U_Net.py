import mindspore.nn as nn
import mindspore.ops as ops
from MIA_Mind import MIA

# -------------------- U-Net with MIA --------------------

class ConvBlock(nn.Cell):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, pad_mode='pad')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, pad_mode='pad')
        self.relu2 = nn.ReLU()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class UNetCBAM(nn.Cell):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetCBAM, self).__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(64, 128)

        self.up1 = nn.Conv2dTranspose(128, 64, 2, stride=2)
        self.cbam1 = MIA(64)
        self.dec1 = ConvBlock(128, 64)

        self.up2 = nn.Conv2dTranspose(64, 32, 2, stride=2)
        self.cbam2 = MIA(32)
        self.dec2 = ConvBlock(64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        up1 = self.cbam1(self.up1(b))
        d1 = self.dec1(self.concat((up1, e2)))

        up2 = self.cbam2(self.up2(d1))
        d2 = self.dec2(self.concat((up2, e1)))

        return self.final(d2)
