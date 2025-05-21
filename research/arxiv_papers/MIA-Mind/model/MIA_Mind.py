import mindspore.nn as nn
import mindspore.ops as ops

# -------------------- MIA-Mind 注意力模块 --------------------

class ChannelAttention(nn.Cell):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.max_pool = ops.ReduceMax(keep_dims=True)
        self.fc1 = nn.Dense(in_channels, in_channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(in_channels // reduction, in_channels)
        self.sigmoid = ops.Sigmoid()

    def construct(self, x):
        b, c, _, _ = x.shape
        avg_pool = self.avg_pool(x, (2, 3)).view((b, c))
        max_pool = self.max_pool(x, (2, 3)).view((b, c))
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        out = self.sigmoid(avg_out + max_out).view((b, c, 1, 1))
        return x * out

class SpatialAttention(nn.Cell):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, pad_mode='pad')
        self.sigmoid = ops.Sigmoid()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.reduce_max = ops.ReduceMax(keep_dims=True)
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        avg_out = self.reduce_mean(x, 1)
        max_out = self.reduce_max(x, 1)
        x_cat = self.concat((avg_out, max_out))
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn

class MIA(nn.Cell):
    def __init__(self, in_channels):
        super(MIA, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def construct(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
