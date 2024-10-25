# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""misc functions for program"""
import os
import datetime as dt
from pathlib import Path
import shutil
import json

from mindspore import context
from mindspore import nn
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src import data
from src.trainer.train_one_step_with_ema import TrainOneStepWithEMA
from src.trainer.train_one_step_with_scale_and_clip_global_norm import \
    TrainOneStepWithLossScaleCellGlobalNormClip

from src.tools.utils import setup_for_distributed


def set_device(device_target, device_id):
    """Set device and ParallelMode(if device_num > 1)"""
    rank = 0
    # set context and device
    device_num = int(os.environ.get("DEVICE_NUM", 1))

    if device_target == "Ascend":
        if device_num > 1:
            context.set_context(device_id=int(os.environ["DEVICE_ID"]))
            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True
            )
            rank = get_rank()
        else:
            context.set_context(device_id=device_id)
    elif device_target == "GPU":
        if device_num > 1:
            init(backend_name='nccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True
            )
            rank = get_rank()
            setup_for_distributed(rank == 0)

        else:
            context.set_context(device_id=device_id)
    else:
        raise ValueError("Unsupported platform.")

    return rank


def get_dataset(args, training=True):
    """"Get model according to args.set"""
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args, training)

    return dataset



def load_pretrained(model, pretrained, num_classes, exclude_epoch_state=True):
    """"Load pretrained weights if pretrained is given"""

    if os.path.isfile(pretrained):
        print("=> loading pretrained weights from '{}'".format(pretrained))
        param_dict = load_checkpoint(pretrained)
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != num_classes:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        if exclude_epoch_state:
            if 'epoch_num' in param_dict:
                param_dict.pop('epoch_num')
            if 'step_num' in param_dict:
                param_dict.pop('step_num')
            if 'learning_rate' in param_dict:
                param_dict.pop('learning_rate')
            if 'global_step' in param_dict:
                param_dict.pop('global_step')

        load_param_into_net(model, param_dict)
    else:
        print("=> no pretrained weights found at '{}'".format(pretrained))


def get_train_one_step(args, net_with_loss, optimizer):
    """get_train_one_step cell"""
    if args.is_dynamic_loss_scale:
        print(f"=> Using DynamicLossScaleUpdateCell")
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(
            loss_scale_value=2 ** 24, scale_factor=2, scale_window=2000
        )
    else:
        print(
            "=> Using FixedLossScaleUpdateCell, "
            f"loss_scale_value:{args.loss_scale}"
        )
        scale_sense = nn.wrap.FixedLossScaleUpdateCell(
            loss_scale_value=args.loss_scale
        )
    if args.model_ema:
        print(f"=> Using EMA. ema_decay: {args.model_ema_decay}")
        net_with_loss = TrainOneStepWithEMA(
            net_with_loss,
            optimizer,
            scale_sense=scale_sense,
            with_ema=args.model_ema,
            ema_decay=args.model_ema_decay)
    elif args.use_clip_grad_norm:
        print(
            "=> Using gradient clipping by norm, clip_grad_norm: "
            f"{args.clip_grad_norm}"
        )
        net_with_loss = TrainOneStepWithLossScaleCellGlobalNormClip(
            net_with_loss,
            optimizer,
            scale_sense,
            use_global_norm=args.use_clip_grad_norm,
            clip_global_norm_value=args.clip_grad_norm
        )
    else:
        print("=> Use simple loss scale.")
        net_with_loss = nn.TrainOneStepWithLossScaleCell(
            net_with_loss, optimizer, scale_sense=scale_sense
        )
    return net_with_loss


def get_directories(
        model_name: str,
        output_dir: str,
        postfix: str = '',
        rank: int = 0,
):
    """Get directories"""
    summary_root_dir = Path(output_dir) / 'summary'
    ckpt_root_dir = Path(output_dir) / 'checkpoints'
    best_ckpt_root_dir = Path(output_dir) / 'best_checkpoints'
    prefix = dt.datetime.now().strftime('%y-%m-%d_%H%M%S')
    dir_name = f'{prefix}_{model_name}_{rank}'
    if postfix != '':
        dir_name = f'{dir_name}_{postfix}'

    summary_dir = summary_root_dir / dir_name
    if summary_dir.exists():
        raise RuntimeError(f'Directory {summary_dir} already exist.')

    ckpt_dir = ckpt_root_dir / dir_name
    if ckpt_dir.exists():
        raise RuntimeError(f'Directory {ckpt_dir} already exist.')

    best_ckpt_dir = best_ckpt_root_dir / dir_name
    if best_ckpt_dir.exists():
        raise RuntimeError(f'Directory {best_ckpt_dir} already exist.')

    summary_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True)
    best_ckpt_dir.mkdir(parents=True)

    return str(summary_dir), str(ckpt_dir), str(best_ckpt_dir)


def save_config(args, dir_to_save):
    """Save config to dir"""
    if args.config is not None:
        shutil.copy(args.config, dir_to_save)
    with open(Path(dir_to_save) / 'all_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
