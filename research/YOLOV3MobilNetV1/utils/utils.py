import os
import random
import yaml
import cv2
from datetime import datetime
import numpy as np

import mindspore as ms
from mindspore import context, ops, Tensor, nn
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode

from utils import logger


def set_seed(seed=2):
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)


def set_default(args):
    # Set Context
    context.set_context(mode=args.ms_mode, device_target=args.device_target, max_call_depth=2000)
    if args.device_target == "Ascend":
        device_id = int(os.getenv("DEVICE_ID", 0))
        context.set_context(device_id=device_id)
    elif args.device_target == "GPU" and args.ms_enable_graph_kernel:
        context.set_context(enable_graph_kernel=True)
    # Set Parallel
    if args.is_parallel:
        init()
        args.rank, args.rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(device_num=args.rank_size, parallel_mode=parallel_mode, gradients_mean=True)
    else:
        args.rank, args.rank_size = 0, 1
    # Set Default
    args.total_batch_size = args.per_batch_size * args.rank_size
    args.sync_bn = args.sync_bn and context.get_context("device_target") == "Ascend" and args.rank_size > 1
    args.accumulate = max(1, np.round(args.nbs / args.total_batch_size)) if args.auto_accumulate else args.accumulate
    # optimizer
    args.optimizer.warmup_epochs = args.optimizer.get("warmup_epochs", 0)
    args.optimizer.min_warmup_step = args.optimizer.get("min_warmup_step", 0)
    args.optimizer.epochs = args.epochs
    args.optimizer.nbs = args.nbs
    args.optimizer.accumulate = args.accumulate
    args.optimizer.total_batch_size = args.total_batch_size
    # data
    cv2.setNumThreads(args.opencv_threads_num)  # Set the number of threads for opencv.
    args.data.nc = 1 if args.single_cls else int(args.data.nc)  # number of classes
    args.data.names = ["item"] if args.single_cls and len(args.names) != 1 else args.data.names  # class names
    assert len(args.data.names) == args.data.nc, "%g names found for nc=%g dataset in %s" % (
        len(args.data.names),
        args.data.nc,
        args.config,
    )
    # Directories and Save run settings
    time = get_broadcast_datetime(rank_size=args.rank_size)
    args.save_dir = os.path.join(
        args.save_dir, f'{time[0]:04d}.{time[1]:02d}.{time[2]:02d}-{time[3]:02d}.{time[4]:02d}.{time[5]:02d}')
    os.makedirs(args.save_dir, exist_ok=True)
    if args.rank % args.rank_size == 0:
        with open(os.path.join(args.save_dir, "cfg.yaml"), "w") as f:
            yaml.dump(vars(args), f, sort_keys=False)

    # callback
    args.callback = args.get('callback', [])

    # Set Logger
    logger.setup_logging(
        logger_name="MindYOLO", log_level=args.log_level, rank_id=args.rank, device_per_servers=args.rank_size
    )
    logger.setup_logging_file(log_dir=os.path.join(args.save_dir, "logs"))


def drop_inconsistent_shape_parameters(model, param_dict):
    updated_param_dict = dict()

    # TODO: hard code
    param_dict = {k.replace('ema.', ''): v for k, v in param_dict.items()}

    for param in model.get_parameters():
        name = param.name
        if name in param_dict:
            if param_dict[name].shape == param.shape:
                updated_param_dict[name] = param_dict[name]
            else:
                logger.warning(
                    f"Dropping checkpoint parameter `{name}` with shape `{param_dict[name].shape}`, "
                    f"which is inconsistent with cell shape `{param.shape}`"
                )
        else:
            logger.warning(f"Cannot find checkpoint parameter `{name}`.")
    return updated_param_dict


def load_pretrain(network, weight, ema=None, ema_weight=None, strict=True):
    if weight.endswith(".ckpt"):
        param_dict = ms.load_checkpoint(weight)
        if not strict:
            param_dict = drop_inconsistent_shape_parameters(network, param_dict)
        ms.load_param_into_net(network, param_dict)
        logger.info(f'Pretrain model load from "{weight}" success.')
    if ema:
        if ema_weight.endswith(".ckpt"):
            param_dict_ema = ms.load_checkpoint(ema_weight)
            if not strict:
                param_dict_ema = drop_inconsistent_shape_parameters(ema.ema, param_dict_ema)
            ms.load_param_into_net(ema.ema, param_dict_ema)
            logger.info(f'Ema pretrain model load from "{ema_weight}" success.')
        else:
            ema.clone_from_model()
            logger.info("ema_weight not exist, default pretrain weight is currently used.")


def freeze_layers(network, freeze=[]):
    if len(freeze) > 0:
        freeze = [f"model.{x}." for x in freeze]  # parameter names to freeze (full or partial)
        for n, p in network.parameters_and_names():
            if any(x in n for x in freeze):
                logger.info("freezing %s" % n)
                p.requires_grad = False


def get_broadcast_datetime(rank_size=1, root_rank=0):
    time = datetime.now()
    time_list = [time.year, time.month, time.day, time.hour, time.minute, time.second, time.microsecond]
    if rank_size <=1:
        return time_list

    # only broadcast in distribution mode
    x = broadcast((Tensor(time_list, dtype=ms.int32),), root_rank)
    x = x[0].asnumpy().tolist()
    return x

@ms.jit
def broadcast(x, root_rank):
    return ops.Broadcast(root_rank=root_rank)(x)

class AllReduce(nn.Cell):
    """
    a wrapper class to make ops.AllReduce become a Cell. This is a workaround for sync_wait
    """
    def __init__(self):
        super(AllReduce, self).__init__()
        self.all_reduce = ops.AllReduce(op=ops.ReduceOp.SUM)

    def construct(self, x):
        return self.all_reduce(x)


class Synchronizer:
    def __init__(self, rank_size=1):
        # this init method should be run only once
        self.all_reduce = AllReduce()
        self.rank_size = rank_size

    def __call__(self):
        if self.rank_size <= 1:
            return
        sync = Tensor(np.array([1]).astype(np.int32))
        sync = self.all_reduce(sync)
        sync = sync.asnumpy()[0]
        if sync != self.rank_size:
            raise ValueError(f'Sync value {sync} is not equal to rank size {self.rank_size}.'
                             f' There might be wrong with devices')
