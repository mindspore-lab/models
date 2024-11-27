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

