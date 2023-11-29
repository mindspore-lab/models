import mindspore as ms
from mindspore import nn, ops

from typing import Type


class MLPBlock(nn.Cell):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Cell] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Dense(embedding_dim, mlp_dim)
        self.lin2 = nn.Dense(mlp_dim, embedding_dim)
        self.act = act()

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Cell):
    def __init__(self, num_channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = ms.Parameter(ops.ones(num_channels))
        self.bias = ms.Parameter(ops.zeros(num_channels))
        self.eps = epsilon

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1).swapaxes(1, 2) # (bs, c, h, w) -> (bs, hw, c)

        u = x.mean(-1, keep_dims=True) # (bs, hw, 1)
        s = (x - u).pow(2).mean(-1, keep_dims=True) # (bs, hw, 1)
        x = (x - u) / ops.sqrt(s + self.eps) # (bs, hw, c)

        x = x.swapaxes(1, 2).reshape(bs, c, h, w)

        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
