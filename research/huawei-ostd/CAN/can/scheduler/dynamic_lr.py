"""Meta learning rate scheduler.

This module implements exactly the same learning rate scheduler as native PyTorch,
see `"torch.optim.lr_scheduler" <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
At present, only `constant_lr`, `linear_lr`, `polynomial_lr`, `exponential_lr`, `step_lr`, `multi_step_lr`,
`cosine_annealing_lr`, `cosine_annealing_warm_restarts_lr` are implemented. The number, name and usage of
the Positional Arguments are exactly the same as those of native PyTorch.

However, due to the constraint of having to explicitly return the learning rate at each step, we have to
introduce additional Keyword Arguments. There are only three Keyword Arguments introduced,
namely `lr`, `steps_per_epoch` and `epochs`, explained as follows:
`lr`: the basic learning rate when creating optim in torch.
`steps_per_epoch`: the number of steps(iterations) of each epoch.
`epochs`: the number of epoch. It and `steps_per_epoch` determine the length of the returned lrs.

Since most scheduler in PyTorch are coarse-grained, that is the learning rate is constant within a single epoch.
For non-stepwise scheduler, we introduce several fine-grained variation, that is the learning rate
is also changed within a single epoch. The function name of these variants have the `refined` keyword.
The implemented fine-grained variation are list as follows: `linear_refined_lr`, `polynomial_refined_lr`, etc.

"""
import math


def linear_lr(start_factor, end_factor, total_iters, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    start_lr = lr * start_factor
    end_lr = lr * end_factor
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        multiplier = min(epoch_idx, total_iters) / total_iters
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def linear_refined_lr(start_factor, end_factor, total_iters, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    start_lr = lr * start_factor
    end_lr = lr * end_factor
    for i in range(steps):
        epoch_idx = i / steps_per_epoch
        multiplier = min(epoch_idx, total_iters) / total_iters
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def cosine_decay_lr(decay_epochs, eta_min, *, eta_max, steps_per_epoch, epochs, num_cycles=1, cycle_decay=1.0):
    """update every epoch"""
    tot_steps = steps_per_epoch * epochs
    lrs = []

    for c in range(num_cycles):
        lr_max = eta_max * (cycle_decay**c)
        delta = 0.5 * (lr_max - eta_min)
        for i in range(steps_per_epoch * decay_epochs):
            t_cur = math.floor(i / steps_per_epoch)
            t_cur = min(t_cur, decay_epochs)
            lr_cur = eta_min + delta * (1.0 + math.cos(math.pi * t_cur / decay_epochs))
            if len(lrs) < tot_steps:
                lrs.append(lr_cur)
            else:
                break

    if epochs > num_cycles * decay_epochs:
        for i in range((epochs - (num_cycles * decay_epochs)) * steps_per_epoch):
            lrs.append(eta_min)

    return lrs


def cosine_decay_refined_lr(decay_epochs, eta_min, *, eta_max, steps_per_epoch, epochs, num_cycles=1, cycle_decay=1.0):
    """update every step"""
    tot_steps = steps_per_epoch * epochs
    lrs = []

    for c in range(num_cycles):
        lr_max = eta_max * (cycle_decay**c)
        delta = 0.5 * (lr_max - eta_min)
        for i in range(steps_per_epoch * decay_epochs):
            t_cur = i / steps_per_epoch
            t_cur = min(t_cur, decay_epochs)
            lr_cur = eta_min + delta * (1.0 + math.cos(math.pi * t_cur / decay_epochs))
            if len(lrs) < tot_steps:
                lrs.append(lr_cur)
            else:
                break

    if epochs > num_cycles * decay_epochs:
        for i in range((epochs - (num_cycles * decay_epochs)) * steps_per_epoch):
            lrs.append(eta_min)

    return lrs
