import os
import time
from datetime import datetime

import mindspore as ms
from mindspore import ops, Tensor, context, ParallelMode, mint
from mindspore.communication import init, get_group_size, get_rank

from segment_anything.utils import logger
from segment_anything.utils.logger import setup_logging


RANK_SIZE = 1
RANK_ID = 0


def freeze_layer(network, specify_prefix=None, filter_prefix=None):
    """freeze layers, default to freeze all the input network"""
    for n, p in network.parameters_and_names():
        if filter_prefix is not None and n.startswith(filter_prefix):
            continue
        if specify_prefix is not None and not n.startswith(specify_prefix):
            continue
        p.requires_grad = False


def reduce_with_mask(input, valid_mask, reduction='mean'):
    if valid_mask is None:
        return mint.sum(input)
    if valid_mask.dtype != input.dtype:
        valid_mask = valid_mask.astype(input.dtype)
    num_valid = mint.sum(valid_mask)
    if reduction == 'mean':
        return mint.sum(input * valid_mask) / num_valid
    else:  # sum
        return mint.sum(input * valid_mask)


def calc_iou(pred_mask: ms.Tensor, gt_mask: ms.Tensor, epsilon=1e-7):
    """
    Args:
        pred_mask (ms.Tensor): prediction mask, with shape (b, n, h, w), 0 for background and 1 for foreground
        gt_mask (ms.Tensor): gt mask, with shape (b, n, h, w), value is 0 or 1.
    """
    hw_dim = (-2, -1)
    intersection = mint.sum(mint.mul(pred_mask, gt_mask), dim=hw_dim)  # (b, n)
    union = mint.sum(pred_mask, dim=hw_dim) + mint.sum(gt_mask, dim=hw_dim) - intersection
    batch_iou = intersection / (union + epsilon)  # (b, n)

    return batch_iou


# decorate collection communication operator with ms.jit to make run in graph mode for compatibility with graph context.
@ms.jit
def broadcast(x, root_rank):
    return ops.Broadcast(root_rank=root_rank)(x)


@ms.jit
def all_reduce(x):
    # calling ops.AllReduce without communication init will raise an error
    if RANK_SIZE <= 1:
        return x
    return ops.AllReduce()(x)

def get_broadcast_datetime(rank_size=1, root_rank=0):
    time = datetime.now()
    time_list = [time.year, time.month, time.day, time.hour, time.minute, time.second, time.microsecond]
    if rank_size <=1:
        return time_list

    # only broadcast in distribution mode
    x = broadcast((Tensor(time_list, dtype=ms.int32),), root_rank)
    x = x[0].asnumpy().tolist()
    return x


def set_distributed(distributed):
    rank_id = 0
    rank_size = 1
    if distributed:
        logger.info('distributed training start')
        init()
        rank_id = get_rank()
        rank_size = get_group_size()
        logger.info(f'current rank {rank_id}/{rank_size}')
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=rank_size, gradients_mean=True,
                                          parallel_mode=ParallelMode.DATA_PARALLEL)
    main_device = rank_id == 0

    # This is the only palace where global rank_id and rank_size can be modified
    global RANK_ID, RANK_SIZE
    RANK_ID, RANK_SIZE= rank_id, rank_size

    print(f'rank {rank_id}/{rank_size}, main_device: {main_device}')

    return rank_id, rank_size, main_device


def update_rank_to_dataloader_config(rank_id, rank_size, args_train_loader, args_eval_loader, arg_callback=None):
    """
    rank_id and rank_size are acquired during runtime, thus they need to updated to config
    """
    if args_train_loader is not None:
        args_train_loader.rank_id = rank_id
        args_train_loader.rank_size = rank_size
    if args_eval_loader is not None:
        args_eval_loader.rank_id = rank_id
        args_eval_loader.rank_size = rank_size

    # a hack implementation to update runtime-defined setting in callback args
    if arg_callback is not None:
        for cb in arg_callback:
            if cb.type.endswith('EvalWhileTrain'):
                cb.data_loader = args_eval_loader



def set_directory_and_log(main_device, rank_id, rank_size, work_root, log_level, args_callback=None):
    """
    generate a time-dependent but device-independent directory name. Creat a working directory.
    also update the generated directory name to callback args.

    Returns:
        directory base name
    """
    unite_time = get_broadcast_datetime(rank_size=rank_size)
    save_dir = f'{unite_time[0]}_{unite_time[1]:02d}_{unite_time[2]:02d}-' \
               f'{unite_time[3]:02d}_{unite_time[4]:02d}_{unite_time[5]:02d}'

    if main_device:
        work_dir = os.path.join(work_root, save_dir)
        os.makedirs(work_dir, exist_ok=True)
    setup_logging(log_dir=os.path.join(work_root, save_dir, 'log'), log_level=log_level, rank_id=rank_id)

    # a hack implementation to update runtime-defined setting in callback args
    if args_callback is not None:
        for cb in args_callback:
            if cb.type.endswith('SaveCkpt'):
                hack_list = {'save_dir': save_dir, 'main_device': main_device, 'work_root': work_root}
                cb.update(hack_list)
    return save_dir


class Timer:
    def __init__(self, name=''):
        self.name = name
        self.start = 0.0
        self.end = 0.0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f'{self.name} cost time {self.end - self.start:.3f}')
