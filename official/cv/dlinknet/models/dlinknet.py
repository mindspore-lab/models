from functools import partial

from mindspore import nn, load_checkpoint, load_param_into_net
from mindspore.common.tensor import Tensor
from mindspore.mint.nn.functional import relu, batch_norm
from mindspore.mint import sigmoid
from models.resnet import resnet34, resnet50, kaiming_uniform
from config import parse_args

conv_weight_init = partial(kaiming_uniform, mode="fan_out", nonlinearity='relu')


class BN2d(nn.BatchNorm2d):
    def construct(self, x):
        return batch_norm(x, self.moving_mean, self.moving_variance, self.gamma, self.beta, training=self.training,
                          momentum=self.momentum, eps=self.eps)


class Dblock_more_dilate(nn.Cell):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1, pad_mode='pad', has_bias=True,
                                 weight_init=Tensor(conv_weight_init((channel, channel, 3, 3))))
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2, pad_mode='pad', has_bias=True,
                                 weight_init=Tensor(conv_weight_init((channel, channel, 3, 3))))
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4, pad_mode='pad', has_bias=True,
                                 weight_init=Tensor(conv_weight_init((channel, channel, 3, 3))))
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8, pad_mode='pad', has_bias=True,
                                 weight_init=Tensor(conv_weight_init((channel, channel, 3, 3))))
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16, pad_mode='pad',
                                 weight_init=Tensor(conv_weight_init((channel, channel, 3, 3))))

    def construct(self, x):
        dilate1_out = relu(self.dilate1(x))
        dilate2_out = relu(self.dilate2(dilate1_out))
        dilate3_out = relu(self.dilate3(dilate2_out))
        dilate4_out = relu(self.dilate4(dilate3_out))
        dilate5_out = relu(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class Dblock(nn.Cell):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1, pad_mode='pad', has_bias=True,
                                 weight_init=Tensor(conv_weight_init((channel, channel, 3, 3))))
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2, pad_mode='pad', has_bias=True,
                                 weight_init=Tensor(conv_weight_init((channel, channel, 3, 3))))
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4, pad_mode='pad', has_bias=True,
                                 weight_init=Tensor(conv_weight_init((channel, channel, 3, 3))))
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8, pad_mode='pad', has_bias=True,
                                 weight_init=Tensor(conv_weight_init((channel, channel, 3, 3))))

    def construct(self, x):
        dilate1_out = relu(self.dilate1(x))
        dilate2_out = relu(self.dilate2(dilate1_out))
        dilate3_out = relu(self.dilate3(dilate2_out))
        dilate4_out = relu(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DecoderBlock(nn.Cell):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1, has_bias=True,
                               weight_init=Tensor(conv_weight_init((in_channels // 4, in_channels, 1, 1))))
        self.norm1 = BN2d(in_channels // 4)
        self.deconv2 = nn.Conv2dTranspose(in_channels // 4, in_channels // 4,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          pad_mode='pad')
        self.norm2 = BN2d(in_channels // 4)
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1, has_bias=True,
                               weight_init=Tensor(conv_weight_init((n_filters, in_channels // 4, 1, 1))))
        self.norm3 = BN2d(n_filters)

    def construct(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = relu(x)
        x = self.deconv2(x)  # Conv2dTranspose
        x = self.norm2(x)
        x = relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = relu(x)
        return x


class DLinkNet34(nn.Cell):
    def __init__(self, num_classes=1, use_backbone=False):
        super(DLinkNet34, self).__init__()
        self.args = parse_args()

        filters = [64, 128, 256, 512]
        resnet = resnet34(1001)
        if use_backbone:
            param_dict = load_checkpoint(self.args.pretrained_ckpt)
            load_param_into_net(resnet, param_dict)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.pad = resnet.pad
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.Conv2dTranspose(filters[0], 32, 4, stride=2, padding=1, pad_mode='pad',
                                               has_bias=True,
                                               weight_init=Tensor(conv_weight_init((filters[0], 32, 4, 4))))
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1, pad_mode='pad',
                                    has_bias=True, weight_init=Tensor(conv_weight_init((32, 32, 3, 3))))
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1, pad_mode='pad',
                                    has_bias=True,
                                    weight_init=Tensor(conv_weight_init((num_classes, 32, 3, 3))))

    def construct(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = relu(x)
        x = self.pad(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = relu(out)
        out = self.finalconv2(out)
        out = relu(out)
        out = self.finalconv3(out)

        return sigmoid(out)


class DLinkNet50(nn.Cell):
    def __init__(self, num_classes=1, use_backbone=False):
        super(DLinkNet50, self).__init__()
        self.args = parse_args()

        filters = [256, 512, 1024, 2048]
        resnet = resnet50(1001)
        if use_backbone:
            param_dict = load_checkpoint(self.args.pretrained_ckpt)
            load_param_into_net(resnet, param_dict)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.Conv2dTranspose(filters[0], 32, 4, stride=2, padding=1, pad_mode='pad',
                                               has_bias=True,
                                               weight_init=Tensor(conv_weight_init((filters[0], 32, 4, 4))))
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1, pad_mode='pad', has_bias=True,
                                    weight_init=Tensor(conv_weight_init((32, 32, 3, 3))))
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1, pad_mode='pad', has_bias=True,
                                    weight_init=Tensor(conv_weight_init((num_classes, 32, 3, 3))))

    def construct(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = relu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = relu(out)
        out = self.finalconv2(out)
        out = relu(out)
        out = self.finalconv3(out)

        return sigmoid(out)
