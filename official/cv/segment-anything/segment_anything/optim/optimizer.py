from typing import List

from mindspore import nn

from segment_anything.optim.group_param import create_group_param
from segment_anything.optim.scheduler import create_lr_scheduler
from segment_anything.utils.registry import OPTIMIZER_REGISTRY


def create_optimizer(
    params,
    args,
    step_per_epoch,
    epoch_size
):
    r"""Creates optimizer by name.

    Args:
        params: network parameters.
        optim: optimizer name like 'sgd', 'adamw', 'momentum'.
        lr: learning rate, float or lr scheduler. Fixed and dynamic learning rate are supported. Default: 1e-3.
        weight_decay: weight decay factor. Default: 0.
        momentum: momentum if the optimizer supports. Default: 0.9.
        nesterov: Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients. Default: False.
        loss_scale: A floating point value for the loss scale, which must be larger than 0.0. Default: 1.0.

    Returns:
        Optimizer object
    """
    optimizer = OPTIMIZER_REGISTRY.instantiate(**args, params=params,
                                               step_per_epoch=step_per_epoch, epoch_size=epoch_size)
    return optimizer


@OPTIMIZER_REGISTRY.registry_module()
class AdamW(nn.optim.Adam):
    def __init__(self, params: List, lr_scheduler, group_param, step_per_epoch, epoch_size, **kwargs):
        if group_param is None:
            group_param = dict()
        params = create_group_param(params, **group_param)
        lr_scheduler_inst = create_lr_scheduler(lr_scheduler, step_per_epoch=step_per_epoch, epoch_size=epoch_size)
        super().__init__(params, lr_scheduler_inst, **kwargs)
