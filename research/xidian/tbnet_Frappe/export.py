# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""TB-Net exporting app."""

# This script should be run directly with 'python <script> <args>'.

import os
import argparse

import numpy as np
from mindspore import load_checkpoint, load_param_into_net, Tensor, export, set_context, PYNATIVE_MODE, GRAPH_MODE

from src.tbnet import TBNet
from tbnet_config import TBNetConfig


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Export TB-Net.')

    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        default='',
        help="json file for TBNet config"
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help="use which checkpoint(.ckpt) file to export"
    )

    parser.add_argument(
        '--device_id',
        type=int,
        required=False,
        default=0,
        help="device id"
    )

    parser.add_argument(
        '--run_mode',
        type=str,
        required=False,
        default='GRAPH',
        choices=['GRAPH', 'PYNATIVE'],
        help="run code by GRAPH mode or PYNATIVE mode"
    )

    parser.add_argument(
        '--file_name',
        type=str,
        default='tbnet',
        help="model name."
    )

    parser.add_argument(
        '--file_format',
        type=str,
        default='MINDIR',
        choices=['MINDIR', 'AIR'],
        help="model format."
    )
    return parser.parse_args()


def export_tbnet():
    """Export pre-trained TBNet model."""
    args = get_args()

    config_path = args.config_path
    ckpt_path = args.checkpoint_path
    if not os.path.exists(config_path):
        raise ValueError("please check the config path.")
    if not os.path.exists(ckpt_path):
        raise ValueError("please check the checkpoint path.")

    set_context(device_id=args.device_id)
    if args.run_mode == 'GRAPH':
        set_context(mode=GRAPH_MODE)
    else:
        set_context(mode=PYNATIVE_MODE)

    cfg = TBNetConfig(config_path)

    network = TBNet(cfg.num_items, cfg.num_references, cfg.num_relations, cfg.embedding_dim)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)

    item = Tensor(np.ones((1,)).astype(np.int))
    rl1 = Tensor(np.ones((1, cfg.per_item_paths)).astype(np.int))
    ref = Tensor(np.ones((1, cfg.per_item_paths)).astype(np.int))
    rl2 = Tensor(np.ones((1, cfg.per_item_paths)).astype(np.int))
    his = Tensor(np.ones((1, cfg.per_item_paths)).astype(np.int))
    inputs = [item, rl1, ref, rl2, his]
    file_name = os.path.realpath(args.file_name)
    export(network, *inputs, file_name=file_name, file_format=args.file_format)
    if not file_name.endswith("."+args.file_format.lower()):
        file_name = f"{file_name}.{args.file_format.lower()}"
    print(f"{file_name} exported.")


if __name__ == '__main__':
    export_tbnet()
