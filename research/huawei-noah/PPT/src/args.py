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

import os
import sys
import argparse
import yaml


from src.tools import parser as _parser


def _add_summary_arguments(parser: argparse.ArgumentParser):
    """Add summary dump parameters"""
    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')

    parser.add_argument('--keep_checkpoint_max', default=10, type=int,
                        help='keep checkpoint max num')

    parser.add_argument(
        '--ckpt_keep_num',
        type=int,
        default=10,
        help='Keep last N checkpoints.'
    )

    parser.add_argument(
        '--ckpt_save_every_step',
        default=0,
        type=int,
        help='Save every N steps. To use saving by time set this to 0 and use '
             '--ckpt_save_every_sec option.'
             'If both are set `step` is preferred.'
    )
    parser.add_argument(
        '--ckpt_save_every_sec',
        default=3600,
        type=int,
        help='Save every N seconds. To use saving by steps set this to 0 and'
             ' use --ckpt_save_every_step option. '
             'If both are set `step` is preferred.'
    )

    parser.add_argument(
        '--collect_input_data', action='store_true',
        help='Flag to control collecting input data during training. '
             'Important: data us collected only if dataset_sink_mode is False.'
    )
    parser.add_argument(
        '--print_loss_every', type=int, default=20,
        help='Print loss every step.'
    )
    parser.add_argument(
        '--summary_loss_collect_freq',
        type=int,
        default=1,
        help='Frequency of collecting loss while training.'
    )


def _add_device_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--device_target', default='GPU',
                        choices=['GPU', 'Ascend', 'CPU'], type=str,
                        help='device to use for training / testing')

    parser.add_argument('--device_id', default=0, type=int, help='device id')
    parser.add_argument('--device_num', default=1, type=int, help='device num')


def _add_optimizer_arguments(parser: argparse.ArgumentParser):
    """Optimizer parameters"""
    parser.add_argument('--opt', default='adamw', type=str,
                        metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float,
                        metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--beta', default=[0.9, 0.999],
                        type=lambda x: [float(a) for a in x.split(',')],
                        help='beta for optimizer')
    parser.add_argument('--clip_grad', type=float, default=None,
                        metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)'
                        )
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')


def _add_model_arguments(parser: argparse.ArgumentParser):
    """Add arguments related to model to parser."""
    parser.add_argument('--model', default='deit_base_patch16_224', type=str,
                        choices=[
                            'deit_base_patch16_224',
                            'deit_base_distilled_patch16_224',
                            'deit_tiny_patch16_224',
                            'deit_small_patch16_224',
                            'deit_tiny_distilled_patch16_224',
                            'deit_small_distilled_patch16_224',
                            'deit_base_patch16_384',
                            'deit_base_distilled_patch16_384'
                        ],
                        help='Name of model to train')

    parser.add_argument('--num_classes', default=1000, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model_ema', action='store_true')

    parser.add_argument('--model_ema_decay', type=float, default=0.99996,
                        help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true',
                        default=False, help='')


def _add_distilation_arguments(parser: argparse.ArgumentParser):
    """Distillation parameters"""
    parser.add_argument('--teacher_model', default='regnety_160', type=str,
                        choices=['regnety_160'],
                        help='Name of teacher model to train '
                             '(default: "regnety_160"'
                        )
    parser.add_argument('--teacher_path', type=str, default='')
    parser.add_argument('--distillation_type', default='none',
                        choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation_alpha', default=0.5, type=float,
                        help="")
    parser.add_argument('--distillation_tau', default=1.0, type=float, help="")


def _add_dataset_arguments(parser: argparse.ArgumentParser):
    """Add arguments related to dataset to parser."""
    # Dataset parameters
    parser.add_argument('--data_path',
                        default='/imagenet/ILSVRC/Data/CLS-LOC/',
                        help='location of data.')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='train',
        help='Name of directory which contains training subset',
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        default='validation_preprocess',
        help='Name of directory which contains validation subset',
    )

    parser.add_argument(
        '--no_dataset_sink_mode',
        action='store_true',
        help='Dataset sink mode.',
    )


def _add_learning_rate_arguments(parser: argparse.ArgumentParser):
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine_lr', type=str,
                        choices=[
                            'constant_lr', 'cosine_lr',
                            'multistep_lr', 'exp_lr'
                        ],
                        metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_noise', type=float, nargs='+', default=None,
                        metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')

    parser.add_argument(
        '--is_dynamic_loss_scale',
        default=1,
        type=int,
        help='Use Dynamic Loss scale update cell.'
    )

    parser.add_argument(
        '--loss_scale',
        default=1024,
        type=int,
        help='The fixed loss scale value used when updating cell with '
             'fixed loss scaling value (not dynamic one).'
    )

    parser.add_argument('--lr_noise_pct', type=float, default=0.67,
                        metavar='PERCENT',
                        help='learning rate noise limit percent '
                             '(default: 0.67)'
                        )
    parser.add_argument('--lr_noise_std', type=float, default=1.0,
                        metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that'
                             ' hit 0 (1e-5)'
                        )

    parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, '
                             'after cyclic schedule ends'
                        )
    parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler '
                             '(default: 10'
                        )
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1,
                        metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument(
        '--use_clip_grad_norm',
        action='store_true',
        help='Whether to use clip grad norm.'
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=5.0,
        help='Clip grad norm value.'
    )


def _add_augmentation_arguments(parser: argparse.ArgumentParser):
    """Add arguments related to augmentations to parser."""

    parser.add_argument('--min_crop', type=float, default=0.7)

    parser.add_argument('--eval_crop_ratio', default=0.875, type=float,
                        help="Crop ratio for evaluation")
    parser.add_argument('--color_jitter', type=float, default=0.3,
                        metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1',
                        help='AutoAugment policy'
                        )

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, '
                             'bicubic default: "pilcubic")'
                        )

    parser.add_argument('--repeated_aug', action='store_true')

    parser.add_argument('--ThreeAugment', action='store_true')  # 3augment

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) '
                             'augmentation split'
                        )

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. '
                             '(default: 0.8)'
                        )
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. '
                             '(default: 1.0)'
                        )

    parser.add_argument('--mixup_off_epoch', type=float, default=0,
                        help='mixup off epoch'
                        )

    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides '
                             'alpha and enables cutmix if set (default: None)'
                        )
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or'
                             ' cutmix when either/both is enabled'
                        )
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix'
                             ' when both mixup and cutmix enabled'
                        )
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch",'
                             '"pair", or "elem"'
                        )


def _add_main_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--mode', default='GRAPH_MODE',
                        choices=['GRAPH_MODE', 'PYNATIVE_MODE'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce_loss', action='store_true')

    parser.add_argument('--unscale_lr', action='store_true')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--amp_level', default='O2',
                        choices=['O0', 'O1', 'O2', 'O3'], help='AMP Level')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--attn_only', action='store_true')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--exclude_epoch_state', action='store_true',
                        help='exclude epoch state and learning rate')


def parse_args():

    parser = argparse.ArgumentParser('DeiT training and evaluation script',
                                     add_help=False)
    parser.add_argument('--config',
                        help='Config file to use (see configs dir)',
                        default=None, required=False)

    _add_main_train_arguments(parser)
    _add_augmentation_arguments(parser)
    _add_learning_rate_arguments(parser)
    _add_dataset_arguments(parser)
    _add_distilation_arguments(parser)
    _add_model_arguments(parser)
    _add_optimizer_arguments(parser)
    _add_summary_arguments(parser)
    _add_device_arguments(parser)

    args = parser.parse_args()
    return args


def get_args():
    """get_config"""
    args = parse_args()
    override_args = _parser.argv_to_vars(sys.argv)

    if args.config is not None:

        yaml_txt = open(args.config).read()

        # override args
        loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)

        for v in override_args:
            loaded_yaml[v] = getattr(args, v)

        print(f'=> Reading YAML config from {args.config}')

        args.__dict__.update(loaded_yaml)
    os.environ['DEVICE_TARGET'] = args.device_target
    if 'DEVICE_NUM' not in os.environ.keys():
        os.environ['DEVICE_NUM'] = str(args.device_num)
    if 'RANK_SIZE' not in os.environ.keys():
        os.environ['RANK_SIZE'] = str(args.device_num)

    return args
