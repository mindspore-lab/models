import os
import argparse
import ast
import traceback
import mindspore as ms
from mindspore import ops, nn
from mindspore.amp import StaticLossScaler, DynamicLossScaler

from src import get_network
from src.train_warpper import TrainOneStepCell
from src.utils import logger
from src.dataset.loader import create_dataloader
from src.utils.config import load_config, Config, merge
from src.utils.common import init_env, clear
from src.utils.callback import RCNNCallback


os.environ["export HCCL_CONNECT_TIMEOUT"] = "600"


def get_args_train(parents=None):
    parser = argparse.ArgumentParser(description="Train", parents=[parents] if parents else [])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(current_dir, "config/faster_rcnn/faster_rcnn_resnet50_fpn_1x.yml"),
        help="Config file path",
    )
    parser.add_argument("--seed", type=int, default=1234, help="runtime seed")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument("--ms_loss_scaler", type=str, default="static", help="train loss scaler, static/dynamic/none")
    parser.add_argument("--ms_loss_scaler_value", type=float, default=256.0, help="static loss scale value")
    parser.add_argument("--num_parallel_workers", type=int, default=8, help="num parallel worker for dataloader")
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--resume_ckpt", type=str, default="", help="pre trained weights path")
    parser.add_argument("--batch_size", type=int, default=2, help="total batch size for all device")
    parser.add_argument("--mix", type=ast.literal_eval, default=True, help="Mix Precision")
    parser.add_argument("--run_eval", type=ast.literal_eval, default=True, help="run eval")
    parser.add_argument("--eval_parallel", type=ast.literal_eval, default=True, help="run eval")
    parser.add_argument("--log_interval", type=int, default=100, help="log interval")
    parser.add_argument("--lr_init", type=float, default=0.02, help="base learning rate")
    parser.add_argument("--save_dir", type=str, default="output", help="save dir")

    # profiling
    parser.add_argument("--run_profilor", type=ast.literal_eval, default=False, help="run profilor")

    # args for ModelArts
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument("--data_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--ckpt_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--train_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--data_dir", type=str, default="/cache/data", help="ModelArts: obs path to dataset folder")
    args, _ = parser.parse_known_args()
    return args


def train(cfg, network, dataset, optimizer, loss_scaler, grad_reducer, eval_dataset=None):
    train_net = TrainOneStepCell(network, optimizer, loss_scaler, grad_reducer, clip_grads=cfg.clip_grads)
    model = ms.Model(train_net)
    cfg.print_pre_epoch = dataset.get_dataset_size() // cfg.log_interval + 1
    cfg.steps_per_epoch = cfg.print_pre_epoch * cfg.log_interval
    cfg.start_epoch = cfg.start_step // cfg.steps_per_epoch
    model.train(
        (cfg.epochs - cfg.start_epoch) * cfg.print_pre_epoch,
        dataset,
        callbacks=RCNNCallback(cfg, network, optimizer, eval_dataset),
        dataset_sink_mode=True,
        sink_size=cfg.log_interval,
    )
    if cfg.enable_modelarts:
        from src.utils.modelarts import sync_data

        sync_data(cfg.save_dir, cfg.train_url)


def get_optimizer(cfg, params, lr):
    def init_group_params(params, weight_decay):
        decay_params = []
        no_decay_params = []

        for param in params:
            if len(param.data.shape) > 1:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params},
            {"order_params": params},
        ]

    weight_decay = cfg.weight_decay
    if cfg.filter_bias_and_bn:
        params = init_group_params(params, weight_decay)
        weight_decay = 0.0

    if cfg.type in ["momentum", "sgd"]:
        opt = nn.Momentum(params, lr, momentum=cfg.momentum, weight_decay=weight_decay, use_nesterov=cfg.nesterov)
        return opt
    raise ValueError(f"Not support {cfg.type}")


def get_lr(cfg):
    lrs = []
    warm_base = cfg.lr_init * cfg.lr.warmup_ratio
    milestone_cout = 0
    milestone_step = cfg.lr.milestones[milestone_cout] * cfg.steps_per_epoch
    for i in range((cfg.epochs + 1) * cfg.steps_per_epoch):
        if i < cfg.lr.warmup_step:
            lr = (cfg.lr_init - warm_base) / cfg.lr.warmup_step * i + warm_base
        elif i < milestone_step:
            lr = cfg.lr_init * (cfg.lr.decay_rate**milestone_cout)
        elif i >= cfg.lr.milestones[-1] * cfg.steps_per_epoch:
            lr = cfg.lr_init * cfg.lr.decay_rate ** len(cfg.lr.milestones)
        else:
            milestone_cout += 1
            milestone_step = cfg.lr.milestones[milestone_cout] * cfg.steps_per_epoch
            lr = cfg.lr_init * (cfg.lr.decay_rate**milestone_cout)
        lrs.append(lr)
    return lrs


if __name__ == "__main__":
    args = get_args_train()
    config, helper, choices = load_config(args.config)
    config = merge(args, config)
    config = Config(config)
    init_env(config)
    logger.info(config)
    network = get_network(config)
    if config.mix:
        network.to_float(ms.float32)
        for _, cell in network.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv2d)):
                cell.to_float(ms.float16)

    seg_size = None if not config.data.is_segment else config.data.seg_size
    dataloader, _ = create_dataloader(
        data_config=config.data,
        task="train",
        per_batch_size=config.batch_size,
        rank=config.rank,
        rank_size=config.rank_size,
        shuffle=True,
        drop_remainder=True,
        num_parallel_worker=getattr(config.data, "num_parallel_workers", 8),
        is_segmentaion=config.data.is_segment,
        seg_size=seg_size,
    )
    steps_per_epoch = dataloader.get_dataset_size()
    config.steps_per_epoch = steps_per_epoch
    eval_dataloader = None
    if config.run_eval:
        eval_dataloader, _ = create_dataloader(
            data_config=config.data,
            task="eval",
            per_batch_size=1,
            rank=config.rank,
            rank_size=config.rank_size,
            shuffle=False,
            drop_remainder=False,
            num_parallel_worker=1,
        )
    optimizer = get_optimizer(config.optimizer, network.trainable_params(), get_lr(config))

    config.start_step = 0
    if os.path.exists(config.resume_ckpt):
        params = ms.load_checkpoint(config.resume_ckpt)
        new_params = {}
        for p in params:
            data = params[p]
            if "bbox_cls" in p:
                if data.shape[0] != config.data.nc + 1:
                    print(f"[WARNING] param {p}'s shape {data.shape} is not match num_class {config.data.nc}")
                    continue
            if "bbox_delta" in p:
                if data.shape[0] != config.data.nc * 4:
                    print(f"[WARNING] param {p}'s shape {data.shape} is not match num_class {config.data.nc}")
                    continue
            new_params[p] = data
        ms.load_param_into_net(network, new_params)
        logger.info(f"success to load pretrained ckpt {config.resume_ckpt}")

    loss_scaler = StaticLossScaler(1.0)
    if config.ms_loss_scaler == "dynamic":
        loss_scaler = DynamicLossScaler(
            scale_value=config.get("ms_loss_scaler_value", 2**16),
            scale_factor=config.get("scale_factor", 2),
            scale_window=config.get("scale_window", 2000),
        )
    elif config.ms_loss_scaler == "static":
        loss_scaler = StaticLossScaler(config.get("ms_loss_scaler_value", 2**10))

    grad_reducer = nn.Identity()
    if config.rank_size > 1:
        mean = ms.context.get_auto_parallel_context("gradients_mean")
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, config.rank_size)
        params_num = len(network.trainable_params())
        ms.set_auto_parallel_context(all_reduce_fusion_config=[params_num // 2, params_num // 3 * 2])
    try:
        train(config, network, dataloader, optimizer, loss_scaler, grad_reducer, eval_dataloader)
    except:
        traceback.print_exc()
    finally:
        clear(config)
