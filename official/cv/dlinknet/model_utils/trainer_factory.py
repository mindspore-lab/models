import logging
from typing import Optional, Union

import mindspore as ms
from mindspore import Tensor, context
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.ops import functional as F
from mindspore.train import DynamicLossScaleManager, FixedLossScaleManager, Model


__all__ = [
    "get_metrics",
    "require_customized_train_step",
    "create_trainer",
]

_logger = logging.getLogger(__name__)


def get_metrics(num_classes):
    if num_classes >= 5:
        metrics = {
            "Top_1_Accuracy": nn.Top1CategoricalAccuracy(),
            "Top_5_Accuracy": nn.Top5CategoricalAccuracy(),
        }
    else:
        metrics = {
            "Top_1_Accuracy": nn.Top1CategoricalAccuracy(),
        }
    return metrics


def add_loss_network(network, loss_fn, amp_level):
    """Add loss network."""

    class WithLossCell(nn.Cell):
        "Wrap loss for amp. Cast network output back to float32"

        def __init__(self, backbone, loss_fn):
            super(WithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn

        def construct(self, data, label):
            out = self._backbone(data)
            label = F.mixed_precision_cast(mstype.float32, label)
            return self._loss_fn(F.mixed_precision_cast(mstype.float32, out), label)

    if amp_level == "O2" or amp_level == "O3":
        network = WithLossCell(network, loss_fn)
    else:
        network = nn.WithLossCell(network, loss_fn)
    return network


def create_trainer(
    network: nn.Cell,
    loss: nn.Cell,
    optimizer: nn.Cell,
    amp_level: str,
    loss_scale_type: str = 'dynamic',
    loss_scale: float = 1.0,
    scale_factor: int = 2,
    scale_window: int = 2000,
    drop_overflow_update: bool = True,
):
    """Create Trainer.

    Args:
        network: The backbone network to train, evaluate or predict.
        loss: The function of calculating loss.
        optimizer: The optimizer for training.
        amp_level: The level of auto mixing precision training.
        loss_scale_type: The type of loss scale.
        loss_scale: The value of loss scale.
        drop_overflow_update: Whether to execute optimizer if there is an overflow.

    Returns:
        mindspore.Model

    """
    if loss_scale < 1.0:
        raise ValueError("Loss scale cannot be less than 1.0!")

    if drop_overflow_update is False and loss_scale_type.lower() == "dynamic":
        raise ValueError("DynamicLossScale ALWAYS drop overflow!")

    mindspore_kwargs = dict(
        network=network,
        loss_fn=loss,
        optimizer=optimizer,
        amp_level=amp_level,
    )
    if loss_scale_type.lower() == "fixed":
        mindspore_kwargs["loss_scale_manager"] = FixedLossScaleManager(
            loss_scale=loss_scale, drop_overflow_update=drop_overflow_update
        )
    elif loss_scale_type.lower() == "dynamic":
        mindspore_kwargs["loss_scale_manager"] = DynamicLossScaleManager(
            init_loss_scale=loss_scale, scale_factor=scale_factor, scale_window=scale_window
        )
    elif loss_scale_type.lower() == "auto":
        # We don't explicitly construct LossScaleManager
        _logger.warning(
            "You are using AUTO loss scale, which means the LossScaleManager isn't explicitly pass in "
            "when creating a mindspore.Model instance. "
            "NOTE: mindspore.Model may use LossScaleManager silently. See mindspore.train.amp for details."
        )
    else:
        raise ValueError(f"Loss scale type only support ['fixed', 'dynamic', 'auto'], but got{loss_scale_type}.")
    model = Model(**mindspore_kwargs)
    return model
