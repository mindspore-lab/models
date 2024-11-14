import mindspore as ms
from mindspore import ops, nn


class WithLossCell(nn.Cell):
    r"""
    Cell with loss function.

    Wraps the network with loss function. This Cell accepts data and label as inputs and
    the computed loss will be returned.

    Args:
        backbone (Cell): The backbone network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
        loss_weight (list): weights for each loss_fn

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a tensor means the loss value, the shape of which is usually :math:`()`.
    """

    def __init__(self, net, loss_fn, loss_weight):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.net = net
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight

    def construct(self, data, label):
        outs = self.net(data)
        loss = 0
        for i, out in enumerate(outs):
            out = ops.interpolate(out, label.shape[-2:], mode="bilinear")
            loss += self.loss_weight[i] * self.loss_fn(out, label)
        return loss
