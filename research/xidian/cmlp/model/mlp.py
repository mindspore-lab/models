import mindspore
from mindspore import nn, ops, Tensor
import numpy as np
from mindspore.common.initializer import Normal
from mindspore.ops.function.math_func import *
import mindspore.nn.probability.distribution as msd
class ConvTokenizer(nn.Cell):
    def __init__(self, embedding_dim=64):
        super(ConvTokenizer, self).__init__()
        self.block = nn.SequentialCell(
            nn.Conv2d(3,
                      embedding_dim // 2,
                      kernel_size=3,
                      stride=2,
                      pad_mode='pad',
                      padding=1
                      ),
            nn.BatchNorm2d(embedding_dim // 2,use_batch_statistics=True),
            nn.ReLU(),
            nn.Conv2d(embedding_dim // 2,
                      embedding_dim // 2,
                      kernel_size=3,
                      stride=1,
                      pad_mode='pad',
                      padding=1,
                      ),
            nn.BatchNorm2d(embedding_dim // 2,use_batch_statistics=True),
            nn.ReLU(),
            nn.Conv2d(embedding_dim // 2,
                      embedding_dim,
                      kernel_size=3,
                      stride=1,
                      pad_mode='pad',
                      padding=1,
                      ),
            nn.BatchNorm2d(embedding_dim,use_batch_statistics=True),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3,
            #              stride=2,
            #              pad_mode = 'same',
            #              )
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    def construct(self, x):
        return self.block(x)

class ConvStage(nn.Cell):
    def __init__(self,
                 num_blocks=2,
                 embedding_dim_in=64,
                 hidden_dim=128,
                 embedding_dim_out=128):
        super(ConvStage, self).__init__()
        self.conv_blocks = nn.CellList()
        for i in range(num_blocks):
            block =nn.SequentialCell(
                nn.Conv2d(embedding_dim_in,
                       hidden_dim,
                       kernel_size=1,
                       stride=1,

                       ),
                nn.BatchNorm2d(hidden_dim,use_batch_statistics=True),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1,pad_mode='pad'),
                nn.BatchNorm2d(hidden_dim,use_batch_statistics=True),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, embedding_dim_in, kernel_size=1, stride=1),
                nn.BatchNorm2d(embedding_dim_in,use_batch_statistics=True),
                nn.ReLU()
            )
            self.conv_blocks.append(block)
        self.downsample = nn.Conv2d(embedding_dim_in,
                                 embedding_dim_out,
                                 kernel_size=3,
                                 stride=2,
                                pad_mode='pad',
                                padding=1,
                                has_bias=True
                                    )

    def construct(self, x):
        for block in self.conv_blocks:
            x = x+block(x)
        return self.downsample(x)

class Mlp(nn.Cell):
    def __init__(self,
                 embedding_dim_in,
                 hidden_dim=None,
                 embedding_dim_out=None,
                 activation=nn.GELU):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim_in
        embedding_dim_out = embedding_dim_out or embedding_dim_in
        self.fc1 = nn.Dense(embedding_dim_in, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Dense(hidden_dim, embedding_dim_out)

    def construct(self, x):
        return self.fc2(self.act(self.fc1(x)))

# def drop_pathme(x, drop_prob: float = 0., training: bool = False):
#     """
#     Obtained from: github.com:rwightman/pytorch-image-models
#     Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     shape=mindspore.int32(shape)
#     random_tensor = keep_prob + ops.UniformReal(shape, dtype=x.dtype, device=x.device)
#     floor = ops.Floor()
#     random_tensor= floor(random_tensor)# binarize
#     # output = x.div(keep_prob) * random_tensor
#     output = ops.Div(x,keep_prob) * random_tensor
#     # output = (x/keep_prob) * random_tensor
#     return output
#
# class DropPathme(nn.Cell):
#     """
#     Obtained from: github.com:rwightman/pytorch-image-models
#     Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#
#     def __init__(self, drop_prob=None):
#         super(DropPathme, self).__init__()
#         self.drop_prob = drop_prob
#
#     def construct(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#

class DropPath(nn.Cell):
    """
    DropPath function with keep prob scale.

    Args:
        drop_prob(float): Drop rate, (0, 1). Default:0.0
        scale_by_keep(bool): Determine whether to scale. Default: True.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.
    """

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - self.drop_prob
        if self.keep_prob == 1.0:
            self.keep_prob = 0.9999
        self.scale_by_keep = scale_by_keep
        self.bernoulli = msd.Bernoulli(probs=self.keep_prob)
        self.div = ops.Div()

    def construct(self, x):
        if self.drop_prob > 0.0 and self.training:
            random_tensor = self.bernoulli.sample((x.shape[0],) + (1,) * (x.ndim - 1))
            if self.keep_prob > 0.0 and self.scale_by_keep:
                random_tensor = self.div(random_tensor, self.keep_prob)
            x = x * random_tensor

        return x
class ConvMLPStage(nn.Cell):
    def __init__(self,
                 embedding_dim,
                 dim_feedforward=2048,
                 stochastic_depth_rate=0.1):
        super(ConvMLPStage, self).__init__()

        self.norm1 = nn.LayerNorm([embedding_dim])
        self.channel_mlp1 = Mlp(embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward)
        self.norm2 = nn.LayerNorm([embedding_dim])
        self.connect = nn.Conv2d(embedding_dim,
                              embedding_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              pad_mode='pad',
                              group=embedding_dim,
                              )
        self.connect_norm = nn.LayerNorm([embedding_dim])
        self.channel_mlp2 = Mlp(embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward)
        self.stochastic_depth_rate=stochastic_depth_rate
        if self.stochastic_depth_rate > 0:
            self.drop_path = DropPath(stochastic_depth_rate)
    def construct(self, src):
        if self.stochastic_depth_rate > 0:
            src = src + self.drop_path(self.channel_mlp1(self.norm1(src)))
            src = self.connect(ops.Transpose()(self.connect_norm(src), (0, 3, 1, 2)))
            src=ops.Transpose()(src, (0, 2, 3, 1))
            src = src + self.drop_path(self.channel_mlp2(self.norm2(src)))
        else:
            src = src + (self.channel_mlp1(self.norm1(src)))
            src = self.connect(ops.Transpose()(self.connect_norm(src), (0, 3, 1, 2)))
            src = ops.Transpose()(src, (0, 2, 3, 1))
            src = src + (self.channel_mlp2(self.norm2(src)))
        return src


class ConvDownsample(nn.Cell):
    def __init__(self, embedding_dim_in, embedding_dim_out):
        super().__init__()
        self.downsample = nn.Conv2d(embedding_dim_in, embedding_dim_out, kernel_size=3, stride=2,
                                 padding=1,pad_mode='pad',has_bias=True)

    def construct(self, x):
        x =  ops.Transpose()(x, (0, 3, 1, 2))
        x = self.downsample(x)
        x=ops.Transpose()(x, (0, 2, 3, 1))
        return x


class BasicStage(nn.Cell):
    def __init__(self,
                 num_blocks,
                 embedding_dims,
                 mlp_ratio=1,
                 stochastic_depth_rate=0.1,
                 downsample=True):
        super(BasicStage, self).__init__()
        self.blocks =  nn.CellList()
        start=Tensor(0, mindspore.float32)
        end=Tensor(stochastic_depth_rate, mindspore.float32)
        linspace = ops.LinSpace()
        dpr =linspace(start,end, num_blocks)
        for i in range(num_blocks):
            block = ConvMLPStage(embedding_dim=embedding_dims[0],
                                 dim_feedforward=int(embedding_dims[0] * mlp_ratio),
                                 stochastic_depth_rate=dpr[i],
                                 )
            self.blocks.append(block)
        self.flag=downsample
        if self.flag:
            self.downsample_mlp = ConvDownsample(embedding_dims[0], embedding_dims[1])


    def construct(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.flag:
            x = self.downsample_mlp(x)
        else:
            x=x
        return x

class ConvMLP(nn.Cell):
    def __init__(self,
                 blocks,
                 dims,
                 mlp_ratios,
                 channels=64,
                 n_conv_blocks=3,
                 classifier_head=True,
                 num_classes=1000,
                 *args, **kwargs):
        super(ConvMLP, self).__init__()
        assert len(blocks) == len(dims) == len(mlp_ratios), \
            f"blocks, dims and mlp_ratios must agree in size, {len(blocks)}, {len(dims)} and {len(mlp_ratios)} passed."

        self.tokenizer = ConvTokenizer(embedding_dim=channels)
        self.conv_stages = ConvStage(n_conv_blocks,
                                     embedding_dim_in=channels,
                                     hidden_dim=dims[0],
                                     embedding_dim_out=dims[0])

        self.stages = nn.CellList()
        for i in range(0, len(blocks)):
            stage = BasicStage(num_blocks=blocks[i],
                               embedding_dims=dims[i:i + 2],
                               mlp_ratio=mlp_ratios[i],
                               stochastic_depth_rate=0.1,
                               downsample=(i + 1 < len(blocks)))
            self.stages.append(stage)
        if classifier_head:
            self.norm = nn.LayerNorm([dims[-1]])
            self.head = nn.Dense(dims[-1], num_classes)
        else:
            self.head = None

        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv1d)):
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.TruncatedNormal(sigma=0.02),
                    cell.weight.shape, cell.weight.dtype))
            elif isinstance(cell, (nn.LayerNorm)):
                cell.gamma.set_data(mindspore.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(mindspore.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))

            elif isinstance(cell, (nn.BatchNorm2d)):
                cell.gamma.set_data(mindspore.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(mindspore.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))

    def construct(self, x):
        x = self.tokenizer(x)
        x = self.conv_stages(x)
        x =  ops.Transpose()(x, (0, 2, 3, 1))
        for stage in self.stages:
            x = stage(x)
        if self.head is None:
            return x
        B, _, _, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.norm(x)
        x = x.mean(axis=1)
        x = self.head(x)
        return x

# X = Tensor(np.ones([1, 3, 224, 224]).astype("float32"))
# model=ConvMLP(blocks=[2, 4, 2], dims=[128, 256, 512], mlp_ratios=[2, 2, 2],
#                      classifier_head=True, channels=64, n_conv_blocks=2,num_classes=10)
# # model=ConvTokenizer(embedding_dim=64)
# a=model(X)
# print(a.shape)





