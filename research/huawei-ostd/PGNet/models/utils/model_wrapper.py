import mindspore.ops as ops
from mindspore import nn
from mindspore.common import dtype as mstype


class NetWithLossWrapper(nn.Cell):
    """
    A universal wrapper for any network with any loss.

    Args:
        net (nn.Cell): network
        loss_fn: loss function
        input_indices: The indices of the data tuples which will be fed into the network.
            If it is None, then the first item will be fed only.
        label_indices: The indices of the data tuples which will be fed into the loss function.
            If it is None, then the remaining items will be fed.
    """
    def __init__(self, net, loss_fn, pred_cast_fp32=False, input_indices=None, label_indices=None):
        super().__init__(auto_prefix=False)
        self._net = net
        self._loss_fn = loss_fn
        self.input_indices = input_indices
        self.label_indices = label_indices
        self.pred_cast_fp32 = pred_cast_fp32
        self.cast = ops.Cast()

    def construct(self, *args):
        """
        Args:
            args (Tuple): contains network inputs, labels (given by data loader)
        Returns:
            loss_val (Tensor): loss value
        """
        if self.input_indices is None:
            pred = self._net(args[0])
        else:
            pred = self._net(*select_inputs_by_indices(args, self.input_indices))

        if self.pred_cast_fp32:
            if isinstance(pred, list) or isinstance(pred, tuple):
                pred = [self.cast(p, mstype.float32) for p in pred]
            else:
                pred = self.cast(pred, mstype.float32)

        if self.label_indices is None:
            loss_val = self._loss_fn(pred, *args[1:])
        else:
            loss_val = self._loss_fn(pred, *select_inputs_by_indices(args, self.label_indices))

        return loss_val

def select_inputs_by_indices(inputs, indices):
    new_inputs = list()
    for x in indices:
        new_inputs.append(inputs[x])
    return new_inputs