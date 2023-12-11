from typing import Union, List

from mindspore import nn, ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from segment_anything.utils import logger

_grad_scale = ops.MultitypeFuncGraph("grad_scale")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(ops.Reciprocal()(scale), ops.dtype(grad))


class NetWithLossWrapper(nn.Cell):
    """
    A universal wrapper for any network with any loss.

    Args:
        net (nn.Cell): network
        loss_fn: loss function
        input_indices: contains two element as follows:
            model_input_indices: The indices of the data tuples which will be fed into the network.
            label_indices: The indices of the data tuples which will be fed into the loss function.
    """

    def __init__(self,
                 net: nn.Cell,
                 loss_fn: nn.Cell,
                 input_columns: List[List[str]],
                 all_columns: List[str],
                 ):
        super().__init__(auto_prefix=False)
        self.net = net
        self.loss_fn = loss_fn
        assert len(input_columns) == 2
        self.input_indices = key2index(input_columns[0], all_columns)
        self.label_indices = key2index(input_columns[1], all_columns)
        self.all_columns = all_columns
        self.input_names = [all_columns[i] for i in self.input_indices]
        self.label_names = [all_columns[i] for i in self.label_indices]

    def construct(self, *args):
        """
        Args:
            args (Tuple): contains network inputs, labels (given by data loader)
        Returns:
            loss_val (Tensor): loss value
        """

        # This is a very ugly workaround due to mindspore's disability of dict setitem and unpacking
        # pred = self.net(**select_inputs_by_indices(args, self.input_indices, self.all_columns, return_type='dict'))
        if len(self.input_names) == 2 and self.input_names[0] == 'image' and self.input_names[1] == 'image_patches':  # text-prompt
            pred = self.net(image=args[self.input_indices[0]], image_patches=args[self.input_indices[1]])
        elif len(self.input_names) == 2 and self.input_names[0] == 'image' and self.input_names[1] == 'boxes':
            pred = self.net(image=args[self.input_indices[0]], boxes=args[self.input_indices[1]])
        else:
            raise NotImplementedError

        if not isinstance(pred, tuple):
            pred = (pred, )
        loss_val = self.loss_fn(*pred, *select_inputs_by_indices(args, self.label_indices))

        # currently nn.TrainOneStepCell does not support loss as a tuple
        if isinstance(loss_val, tuple):
            loss_val = loss_val[0]

        return loss_val


def select_inputs_by_indices(inputs, indices, columns_names=None, return_type='tuple'):
    if return_type == 'dict':
        assert columns_names is not None
        assert len(inputs) == len(columns_names)
        new_inputs = dict()
        for i in indices:
            new_inputs[columns_names[i]] = inputs[i]
    elif return_type == 'tuple':
        new_inputs = list()
        for x in indices:
            new_inputs.append(inputs[x])
    else:
        raise NotImplementedError

    return new_inputs

def key2index(keys: Union[List[str], str], all_keys: List[str])-> Union[List[int], int]:
    """
    map keys to index of a list. Usually used in converting model and loss input str to index.
    """
    is_single_element = False
    if isinstance(keys, str):
        is_single_element = True
        keys = [keys]

    index = []
    for k in keys:
        assert k in all_keys
        ind = all_keys.index(k)
        index.append(ind)

    if is_single_element:
        index = index[0]

    return index


class TrainOneStepCellWrapper(nn.TrainOneStepWithLossScaleCell):
    """
    Network training package class with gradient clip.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        grad_clip (bool): Whether clip gradients. Default value is False.
    """

    def __init__(self,
                 network,
                 optimizer,
                 scale_sense=1,
                 grad_clip=False,
                 clip_value=0.1,
                 drop_overflow_update=True,  # do not update when overflow
                 ema=None,
                 ):
        if isinstance(scale_sense, (int, float)):
            scale_sense = nn.FixedLossScaleUpdateCell(scale_sense)
        super(TrainOneStepCellWrapper, self).__init__(network, optimizer, scale_sense)
        self.grad_clip = grad_clip if grad_clip is not None else False
        self.grad_clip_value = clip_value
        self.drop_overflow_update = drop_overflow_update if drop_overflow_update is not None else True
        self.ema = ema

    def construct(self, *inputs):
        weights = self.weights

        loss = self.network(*inputs)  # loss_items may contain both loss and auxiliary value

        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        # loss*grad_scale -> backpropagation -> grad/grad_scale
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)

        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads, clip_norm=self.grad_clip_value)

        grads = self.grad_reducer(grads)

        cond = self.get_overflow_status(status, grads)

        overflow = self.process_loss_scale(cond)

        if overflow:
            print(f'gradient overflow')

        if (not self.drop_overflow_update) or (not overflow):
            self.optimizer(grads)
            if self.ema is not None:
                self.ema.ema_update()

        return loss
