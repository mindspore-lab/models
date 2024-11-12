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
"""TB-Net evaluation app."""

# This script should be run directly with 'python <script> <args>'.

import os
import argparse

from mindspore import Model, load_checkpoint, load_param_into_net, set_context, PYNATIVE_MODE, GRAPH_MODE

from src.tbnet import TBNet, EvalNet
from src.dataset import create_dataset
from src.metrics import AUC, ACC
from tbnet_config import TBNetConfig


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Eval TB-Net.')

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='douban',
        help="'steam' dataset is supported currently"
    )

    parser.add_argument(
        '--csv',
        type=str,
        required=False,
        default='test.csv',
        help="the evaluation csv datafile inside the dataset folder (e.g. test.csv)"
    )


    parser.add_argument(
        '--device_id',
        type=int,
        required=False,
        default=2,
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

    return parser.parse_args()


def eval_tbnet():
    """Evaluation process."""
    args = get_args()

    home = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(home, 'data', args.dataset, 'config.json')
    test_csv_path = os.path.join(home, 'data', args.dataset, args.csv)
    ckpt_path = os.path.join(home, 'checkpoints', args.dataset, f'{args.dataset}.ckpt')

    set_context(device_id=args.device_id)
    if args.run_mode == 'GRAPH':
        set_context(mode=GRAPH_MODE)
    else:
        set_context(mode=PYNATIVE_MODE)

    print(f"creating dataset from {test_csv_path}...")
    cfg = TBNetConfig(config_path)
    eval_ds = create_dataset(test_csv_path, cfg.per_item_paths).batch(cfg.batch_size)

    print(f"creating TBNet from checkpoint {args.checkpoint_id} for evaluation...")
    network = TBNet(cfg.num_items, cfg.num_references, cfg.num_relations, cfg.embedding_dim)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)

    eval_net = EvalNet(network)
    model = Model(network=network, eval_network=eval_net, metrics={'auc': AUC(), 'acc': ACC()})

    print("evaluating...")
    e_out = model.eval(eval_ds, dataset_sink_mode=False)
    print(f'Test AUC:{e_out ["auc"]} ACC:{e_out ["acc"]}')


if __name__ == '__main__':
    eval_tbnet()
