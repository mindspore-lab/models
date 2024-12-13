import math
import numpy as np
from functools import partial
import mindcv

from mindspore import nn, load_checkpoint, load_param_into_net
from mindspore.common.tensor import Tensor
from mindspore.mint.nn.functional import relu, batch_norm
from mindspore.mint import sigmoid
from config import parse_args


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)


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
        if use_backbone:
            resnet = mindcv.create_model('resnet34', pretrained=False, checkpoint_path=self.args.pretrained_ckpt)
        else:
            resnet = mindcv.create_model('resnet34', pretrained=False, checkpoint_path='')

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstmaxpool = resnet.max_pool
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
        if use_backbone:
            resnet = mindcv.create_model('resnet50', pretrained=False, checkpoint_path=self.args.pretrained_ckpt)
        else:
            resnet = mindcv.create_model('resnet50', pretrained=False, checkpoint_path='')

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstmaxpool = resnet.max_pool
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
