import numpy as np

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Parameter, Tensor, nn, ops, jit

from utils import logger
from model.utils import meshgrid


class YOLOv3Head(nn.Cell):
    """
    YOLOv3 Detect Head, convert the output result to a prediction box based on the anchor point.
    """

    def __init__(self, nc=80, anchors=(), stride=(), ch=()):  # detection layer
        super(YOLOv3Head, self).__init__()

        assert isinstance(anchors, (tuple, list)) and len(anchors) > 0
        assert isinstance(stride, (tuple, list)) and len(stride) > 0
        assert isinstance(ch, (tuple, list)) and len(ch) > 0

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        # anchor preprocess
        anchors = np.array(anchors)
        stride = np.array(stride)
        anchors, anchor_grid = self._check_anchor_order(
            anchors=anchors.reshape((self.nl, -1, 2)),
            anchor_grid=anchors.reshape((self.nl, 1, -1, 1, 1, 2)),
            stride=stride,
        )

        self.stride = Tensor(stride, ms.int32)
        self.anchor_grid = Parameter(Tensor(anchor_grid, ms.float32), requires_grad=False)  # shape(nl,1,na,1,1,2)

        self.yolo_output = nn.CellList(
            [nn.Conv2d(x, self.no * self.na, 1, pad_mode="valid", has_bias=True) for x in ch]
        )  # output conv

    def construct(self, x):
        z = ()  # inference output
        outs = ()
        for i in range(self.nl):
            out = self.yolo_output[i](x[i])  # conv
            bs, _, ny, nx = out.shape  # (bs,255,20,20)
            out = out.view(bs, self.na, self.no, ny, nx).transpose((0, 1, 3, 4, 2))  # (bs,3,20,20,85)
            outs += (out,)

            if not self.training:  # inference
                grid_tensor = self._make_grid(nx, ny, out.dtype)
                out[..., 0:2] = (ops.Sigmoid()(out[..., 0:2]) + grid_tensor) / out.shape[2:4]  # xy
                out[..., 2:4] = ops.Exp()(out[..., 2:4]) * self.anchor_grid[i] / self.stride[i] / out.shape[2:4]  # wh
                out[..., 4:] = ops.Sigmoid()(out[..., 4:])
                z += (out.view(bs, -1, self.no),)

        return outs if self.training else (ops.concat(z, 1), outs)

    @staticmethod
    def _make_grid(nx=20, ny=20, dtype=ms.float32):
        # FIXME: Not supported on a specific model of machine
        xv, yv = meshgrid((mnp.arange(nx), mnp.arange(ny)))
        return ops.cast(ops.stack((xv, yv), 2).view((1, 1, ny, nx, 2)), dtype)

    @staticmethod
    def _check_anchor_order(anchors, anchor_grid, stride):
        # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
        a = np.prod(anchor_grid, -1).reshape((-1,))  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = stride[-1] - stride[0]  # delta s
        if np.sign(da) != np.sign(ds):  # same order
            logger.warning("Reversing anchor order")
            anchors = anchors[::-1, ...]
            anchor_grid = anchor_grid[::-1, ...]
        return anchors, anchor_grid
