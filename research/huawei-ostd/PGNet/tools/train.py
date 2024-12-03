import logging
import os
import shutil
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from tools.arg_parser import parse_args_and_config

args, config = parse_args_and_config()

import yaml
from addict import Dict

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

from models.data import build_dataset
from models.losses import build_loss
from models.metrics import build_metric
from models.networks import build_model
from models.optim import create_group_params, create_optimizer
from models.postprocess import build_postprocess
from models.scheduler import create_scheduler
from models.utils.callbacks import EvalSaveCallback
from models.utils.checkpoint import resume_train_network
from models.utils.ema import EMA
from models.utils.logger import set_logger
from models.utils.loss_scaler import get_loss_scales
from models.utils.model_wrapper import NetWithLossWrapper
from models.utils.seed import set_seed
from models.utils.train_step_wrapper import TrainOneStepWrapper

logger = logging.getLogger("mindocr.train")


def main(cfg):
    ms.set_context(mode=cfg.system.mode)
    if cfg.train.max_call_depth:
        ms.set_context(max_call_depth=cfg.train.max_call_depth)
    if cfg.system.mode == 0:
        ms.set_context(jit_config={"jit_level": "O2"})
    if cfg.system.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
        )
        set_logger(
            name="mindocr",
            output_dir=cfg.train.ckpt_save_dir,
            rank=rank_id,
            log_level=eval(cfg.system.get("log_level", "logging.INFO")),
        )
    else:
        device_num = None
        rank_id = None

        set_logger(
            name="mindocr",
            output_dir=cfg.train.ckpt_save_dir,
            rank=0,
            log_level=eval(cfg.system.get("log_level", "logging.INFO")),
        )
        if "DEVICE_ID" in os.environ:
            logger.info(
                f"Standalone training. Device id: {os.environ.get('DEVICE_ID')}, "
                f"specified by environment variable 'DEVICE_ID'."
            )
        else:
            device_id = cfg.system.get("device_id", 0)
            ms.set_context(device_id=device_id)
            logger.info(
                f"Standalone training. Device id: {device_id}, "
                f"specified by system.device_id in yaml config file or is default value 0."
            )

    set_seed(cfg.system.seed)

    # create dataset
    loader_train = build_dataset(
        cfg.train.dataset,
        cfg.train.loader,
        num_shards=device_num,
        shard_id=rank_id,
        is_train=True,
    )
    num_batches = loader_train.get_dataset_size()

    loader_eval = None
    if cfg.system.val_while_train:
        loader_eval = build_dataset(
            cfg.eval.dataset,
            cfg.eval.loader,
            num_shards=device_num,
            shard_id=rank_id,
            is_train=False,
            refine_batch_size=cfg.system.get("refine_batch_size", True),
        )

    # create model
    amp_level = cfg.system.get("amp_level", "O0")
    if ms.get_context("device_target") == "GPU" and cfg.system.val_while_train and amp_level == "O3":
        logger.warning(
            "Model evaluation does not support amp_level O3 on GPU currently. "
            "The program has switched to amp_level O2 automatically."
        )
        amp_level = "O2"
    network = build_model(cfg.model, ckpt_load_path=cfg.model.pop("pretrained", None), amp_level=amp_level)
    num_params = sum([param.size for param in network.get_parameters()])
    num_trainable_params = sum([param.size for param in network.trainable_params()])

    # create loss
    loss_fn = build_loss(cfg.loss.pop("name"), **cfg["loss"])

    net_with_loss = NetWithLossWrapper(
        network,
        loss_fn,
        input_indices=cfg.train.dataset.pop("net_input_column_index", None),
        label_indices=cfg.train.dataset.pop("label_column_index", None),
        pred_cast_fp32=cfg.train.pop("pred_cast_fp32", amp_level != "O0"),
    )

    # get loss scale setting for mixed precision training
    loss_scale_manager, optimizer_loss_scale = get_loss_scales(cfg)

    # build lr scheduler
    lr_scheduler = create_scheduler(num_batches, **cfg["scheduler"])

    # build optimizer
    cfg.optimizer.update({"lr": lr_scheduler, "loss_scale": optimizer_loss_scale})
    params = create_group_params(network.trainable_params(), **cfg.optimizer)
    optimizer = create_optimizer(params, **cfg.optimizer)

    # resume ckpt
    start_epoch = 0
    if cfg.model.resume:
        resume_ckpt = (
            os.path.join(cfg.train.ckpt_save_dir, "train_resume.ckpt")
            if isinstance(cfg.model.resume, bool)
            else cfg.model.resume
        )
        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(network, optimizer, resume_ckpt)
        loss_scale_manager.loss_scale_value = loss_scale
        if cfg.loss_scaler.type == "dynamic":
            loss_scale_manager.cur_iter = cur_iter
            loss_scale_manager.last_overflow_iter = last_overflow_iter

    # build train step cell
    gradient_accumulation_steps = cfg.train.get("gradient_accumulation_steps", 1)
    clip_grad = cfg.train.get("clip_grad", False)
    use_ema = cfg.train.get("ema", False)
    ema = EMA(network, ema_decay=cfg.train.get("ema_decay", 0.9999), updates=0) if use_ema else None

    train_net = TrainOneStepWrapper(
        net_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scale_manager,
        drop_overflow_update=cfg.system.drop_overflow_update,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        clip_norm=cfg.train.get("clip_norm", 1.0),
        ema=ema,
    )

    # build postprocess and metric
    postprocessor = None
    metric = None
    if cfg.system.val_while_train:
        # postprocess network prediction
        postprocessor = build_postprocess(cfg.postprocess)
        metric = build_metric(cfg.metric, device_num=device_num, save_dir=cfg.train.ckpt_save_dir)

    # build callbacks
    eval_cb = EvalSaveCallback(
        network,
        loader_eval,
        postprocessor=postprocessor,
        metrics=[metric],
        pred_cast_fp32=cfg.eval.pop("pred_cast_fp32", amp_level != "O0"),
        rank_id=rank_id,
        device_num=device_num,
        batch_size=cfg.train.loader.batch_size,
        ckpt_save_dir=cfg.train.ckpt_save_dir,
        main_indicator=cfg.metric.main_indicator,
        ema=ema,
        loader_output_columns=cfg.eval.dataset.output_columns,
        input_indices=cfg.eval.dataset.pop("net_input_column_index", None),
        label_indices=cfg.eval.dataset.pop("label_column_index", None),
        meta_data_indices=cfg.eval.dataset.pop("meta_data_column_index", None),
        val_interval=cfg.system.get("val_interval", 1),
        val_start_epoch=cfg.system.get("val_start_epoch", 1),
        log_interval=cfg.system.get("log_interval", 100),
        ckpt_save_policy=cfg.system.get("ckpt_save_policy", "top_k"),
        ckpt_max_keep=cfg.system.get("ckpt_max_keep", 10),
        start_epoch=start_epoch,
    )

    # save args used for training
    if rank_id in [None, 0]:
        with open(os.path.join(cfg.train.ckpt_save_dir, "args.yaml"), "w") as f:
            yaml.safe_dump(cfg.to_dict(), stream=f, default_flow_style=False, sort_keys=False)

    num_devices = device_num if device_num is not None else 1
    global_batch_size = cfg.train.loader.batch_size * num_devices * gradient_accumulation_steps
    model_name = (
        cfg.model.name
        if "name" in cfg.model
        else f"{cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}"
    )
    info_seg = "=" * 40
    logger.info(
        f"\n{info_seg}\n"
        f"Distribute: {cfg.system.distribute}\n"
        f"Model: {model_name}\n"
        f"Total number of parameters: {num_params}\n"
        f"Total number of trainable parameters: {num_trainable_params}\n"
        f"Data root: {cfg.train.dataset.dataset_root}\n"
        f"Optimizer: {cfg.optimizer.opt}\n"
        f"Weight decay: {cfg.optimizer.weight_decay} \n"
        f"Batch size: {cfg.train.loader.batch_size}\n"
        f"Num devices: {num_devices}\n"
        f"Gradient accumulation steps: {gradient_accumulation_steps}\n"
        f"Global batch size: {cfg.train.loader.batch_size}x{num_devices}x{gradient_accumulation_steps}="
        f"{global_batch_size}\n"
        f"LR: {cfg.scheduler.lr} \n"
        f"Scheduler: {cfg.scheduler.scheduler}\n"
        f"Steps per epoch: {num_batches}\n"
        f"Num epochs: {cfg.scheduler.num_epochs}\n"
        f"Clip gradient: {clip_grad}\n"
        f"EMA: {use_ema}\n"
        f"AMP level: {amp_level}\n"
        f"Loss scaler: {cfg.loss_scaler}\n"
        f"Drop overflow update: {cfg.system.drop_overflow_update}\n"
        f"{info_seg}\n"
        f"\nStart training... (The first epoch takes longer, please wait...)\n"
    )
    model = ms.Model(train_net)
    model.train(
        cfg.scheduler.num_epochs,
        loader_train,
        callbacks=[eval_cb],
        dataset_sink_mode=cfg.train.dataset_sink_mode,
        initial_epoch=start_epoch,
    )


if __name__ == "__main__":
    config = Dict(config)
    ckpt_save_dir = config.train.ckpt_save_dir
    os.makedirs(ckpt_save_dir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(ckpt_save_dir, "train_config.yaml"))
    # main train and eval
    main(config)