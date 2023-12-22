import numpy as np


__all__ = ["create_group_param"]


def create_group_param(params, gp_weight_decay=0.0, **kwargs):
    """
    Create group parameters for optimizer.

    Args:
        params: Network parameters
        gp_weight_decay: Weight decay. Default: 0.0
        **kwargs: Others
    """
    if "group_param" in kwargs:
        return group_param_yolov3(params, weight_decay=gp_weight_decay, **kwargs)
    else:
        return params


def group_param_yolov3(
    params,
    weight_decay,
    milestones,
    gamma,
    lr_init,
    warmup_epochs,
    min_warmup_step,
    accumulate,
    epochs,
    steps_per_epoch,
    total_batch_size,
    **kwargs
):
    # old: # weight, gamma, bias/beta
    # new: # bias/beta, weight, others
    pg0, pg1, pg2 = _group_param_common3(params)

    lr_pg0, lr_pg1, lr_pg2 = [], [], []
    lrs = []
    lrs.extend([lr_init] * milestones[0] * steps_per_epoch)
    for i in range(len(milestones)-1):
        lrs.extend([lr_init * gamma] * (milestones[i+1]-milestones[i]) * steps_per_epoch)
        gamma *= gamma
    lrs.extend([lr_init * gamma] * (epochs-milestones[-1]) * steps_per_epoch)

    warmup_steps = max(round(warmup_epochs * steps_per_epoch), min_warmup_step)
    xi = [0, warmup_steps]
    for i in range(epochs * steps_per_epoch):
        _lr = lrs[i]
        if i < warmup_steps:
            lr_pg0.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg1.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg2.append(np.interp(i, xi, [0.0, _lr]))
        else:
            lr_pg0.append(_lr)
            lr_pg1.append(_lr)
            lr_pg2.append(_lr)

    nbs = 64
    weight_decay *= total_batch_size * accumulate / nbs  # scale weight_decay
    group_params = [
        {"params": pg0, "lr": lr_pg0},
        {"params": pg1, "lr": lr_pg1, "weight_decay": weight_decay},
        {"params": pg2, "lr": lr_pg2},
    ]
    return group_params


def _group_param_common3(params):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for p in params:
        if "bias" in p.name or "beta" in p.name:
            pg0.append(p)
        elif "weight" in p.name:
            pg1.append(p)
        else:
            pg2.append(p)

    return pg0, pg1, pg2  # bias/beta, weight, others
