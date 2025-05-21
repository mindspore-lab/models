
from mindvision.classification.models import resnet18
from mindspore import nn
import mindspore as ms
import mindspore
from mindspore import nn, ops
from mindspore import Parameter

class OURS(nn.Cell):
    def __init__(self, num_class=7, num_head=8, pretrained=True):
        super(OURS, self).__init__()
        self.num_head = num_head
        self.resnet = resnet18(pretrained=True)
        if pretrained:
            self.load_pretrained_resnet('./ckpt/Resnet18_mindspore_TKFNet.ckpt')
        self.features = nn.SequentialCell(*list(self.resnet.cells())[:-2])
        for param in self.features.trainable_params():
            param.requires_grad = True

        self.classifier = nn.SequentialCell(
            nn.Dense(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dense(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dense(128, num_class)
        )

        self.gap = ops.AdaptiveAvgPool2D(output_size=1)
        self.flatten = nn.Flatten()

        self.ca = ChannelAttention()
        self.sa = SpatialAttention()
        self.sm = SMFA(512)

    def load_pretrained_resnet(self, ckpt_file):
        pretrained_dict = ms.load_checkpoint(ckpt_file)
        model_dict = self.resnet.parameters_dict()
        for key in pretrained_dict:
            if key in model_dict:
                model_dict[key].set_data(pretrained_dict[key].data)

    def construct(self, x):
        x1 = self.features(x)
        x1 = self.sm(x1)
        x1 = self.ca(x1)
        x1 = self.sa(x1)

        x1 = self.gap(x1)
        x1 = self.flatten(x1)
        out = self.classifier(x1)
        return out



import math


class DMlp(nn.Cell):
    def __init__(self, dim, growth_rate=2.0):
        super(DMlp, self).__init__()
        hidden_dim = int(dim * growth_rate)
        self.depthwise_conv = nn.Conv2d(dim, hidden_dim, kernel_size=3, stride=1, padding=1, group=dim, has_bias=True,
                                        pad_mode='pad')
        self.pointwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, has_bias=True)
        self.act = nn.GELU()
        self.final_conv = nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, has_bias=True)

    def construct(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.act(x)
        x = self.final_conv(x)
        return x


class SMFA(nn.Cell):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, kernel_size=1, has_bias=True)
        self.linear_1 = nn.Conv2d(dim, dim, kernel_size=1, has_bias=True)
        self.linear_2 = nn.Conv2d(dim, dim, kernel_size=1, has_bias=True)

        self.lde = DMlp(dim, 2.0)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, group=dim, has_bias=True, pad_mode='pad')

        self.gelu = nn.GELU()
        self.down_scale = 8

        # 可学习参数
        self.alpha = Parameter(ops.Ones()((1, dim, 1, 1), mindspore.float32))
        self.belt = Parameter(ops.Zeros()((1, dim, 1, 1), mindspore.float32))

        # ops 定义
        self.chunk = ops.Split(axis=1, output_num=2)
        self.interpolate = ops.ResizeNearestNeighbor((1, 1))

    def construct(self, f):
        b, c, h, w = f.shape
        y, x = self.chunk(self.linear_0(f))
        pool_h, pool_w = max(h // self.down_scale, 1), max(w // self.down_scale, 1)
        x_s = self.dw_conv(ops.ResizeNearestNeighbor((pool_h, pool_w))(x))

        x_v = mindspore.numpy.var(x, (-2, -1), keepdims=True)

        scale = x_s * self.alpha + x_v * self.belt
        scale = self.gelu(self.linear_1(scale))

        x_l = x * ops.ResizeNearestNeighbor((h, w))(scale)

        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)


class ChannelAttention(nn.Cell):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.gap = ops.AdaptiveAvgPool2D(output_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(512, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(32, 512)
        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()
        self.mul = ops.Mul()

    def construct(self, x):
        y = self.gap(x)
        y = self.flatten(y)
        y = self.fc1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = self.reshape(y, (x.shape[0], x.shape[1], 1, 1))
        out = self.mul(x, y)
        return out


class SpatialAttention(nn.Cell):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.max_pool = ops.ReduceMax(keep_dims=True)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, pad_mode='pad', padding=padding, has_bias=False)
        self.sigmoid = nn.Sigmoid()
        self.concat = ops.Concat(axis=1)
        self.mul = ops.Mul()

    def construct(self, x):
        avg_out = self.avg_pool(x, (1,))
        max_out = self.max_pool(x, (1,))
        combined = self.concat((avg_out, max_out))
        attn = self.sigmoid(self.conv(combined))
        return self.mul(x, attn)





