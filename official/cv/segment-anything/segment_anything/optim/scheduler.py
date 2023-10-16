from typing import Union, List, Tuple, Dict

from mindspore import ops
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
import mindspore as ms

from segment_anything.utils.registry import LR_SCHEDULER_REGISTRY


def create_lr_scheduler(args: Dict):
    """
    instantiate learning rate scheduler class
    """
    scheduler = LR_SCHEDULER_REGISTRY.instantiate(**args)
    return scheduler


@LR_SCHEDULER_REGISTRY.registry_module()
class SAMDynamicDecayLR(LearningRateSchedule):
    def __init__(self,
                 learning_rate: float,
                 warmup_steps: int,
                 decay_steps: Union[List, Tuple],
                 decay_factor: float,
                 **kwargs,
                 ):
        super(SAMDynamicDecayLR, self).__init__()
        self.lr = learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_step0 = decay_steps[0]  # a workaround for the disability that getitem method cannot be used in constuct
        self.decay_step1 = decay_steps[1]
        self.decay_factor = decay_factor

    def construct(self, global_step):
        factor = self.lr_factor(step=global_step)
        return self.lr * factor

    def lr_factor(self, step):
        step = ops.cast(step, ms.float32)  # cast to int for precise comparison
        if step < self.warmup_steps:
            return step / float(self.warmup_steps)
        elif step < self.decay_step0:
            return ms.Tensor(1.0, dtype=ms.float32)
        elif step < self.decay_step1:
            return ms.Tensor(1 / self.decay_factor, dtype=ms.float32)
        else:
            return ms.Tensor(1 / (self.decay_factor**2), dtype=ms.float32)
