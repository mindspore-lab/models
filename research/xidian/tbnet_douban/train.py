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
"""TB-Net training app."""

# This script should be run directly with 'python <script> <args>'.

import os
import argparse

import numpy as np
from mindspore import Model, Tensor, set_context, PYNATIVE_MODE, GRAPH_MODE
from mindspore.train.serialization import save_checkpoint
from mindspore.train.callback import Callback, TimeMonitor

from src.tbnet import TBNet, NetWithLossCell, TrainStepWrapCell, EvalNet
from src.dataset import create_dataset
from src.metrics import AUC, ACC
from tbnet_config import TBNetConfig


class MyLossMonitor(Callback):
    """My loss monitor definition."""

    def on_train_epoch_end(self, run_context):
        """Print loss at each epoch end."""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())
        print('loss:' + str(loss))

    def on_eval_epoch_end(self, run_context):
        self.on_train_epoch_end(run_context)


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Train TB-Net.')

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='douban',
        help="'steam' dataset is supported currently"
    )

    parser.add_argument(
        '--train_csv',
        type=str,
        required=False,
        default='train.csv',
        help="the train csv datafile inside the dataset folder"
    )

    parser.add_argument(
        '--test_csv',
        type=str,
        required=False,
        default='test.csv',
        help="the test csv datafile inside the dataset folder"
    )

    parser.add_argument(
        '--device_id',
        type=int,
        required=False,
        default=0,
        help="device id"
    )

    parser.add_argument(
        '--epochs',
        type=int,
        required=False,
        default=20,
        help="number of training epochs"
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


def train_tbnet():
    """Training process."""
    args = get_args()

    home = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(home, 'data', args.dataset, 'config.json')
    train_csv_path = os.path.join(home, 'data', args.dataset, args.train_csv)
    test_csv_path = os.path.join(home, 'data', args.dataset, args.test_csv)
    ckpt_dir_path = os.path.join(home, 'checkpoints', args.dataset)

    set_context(device_id=args.device_id)
    if args.run_mode == 'GRAPH':
        set_context(mode=GRAPH_MODE)
    else:
        set_context(mode=PYNATIVE_MODE)

    os.makedirs(ckpt_dir_path, exist_ok=True)

    print(f"creating dataset from {train_csv_path}...")
    cfg = TBNetConfig(config_path)
    train_ds = create_dataset(train_csv_path, cfg.per_item_paths).batch(cfg.batch_size)
    test_ds = create_dataset(test_csv_path, cfg.per_item_paths).batch(cfg.batch_size)

    print("creating TBNet for training...")
    network = TBNet(cfg.num_items, cfg.num_references, cfg.num_relations, cfg.embedding_dim)
    loss_net = NetWithLossCell(network, cfg.kge_weight, cfg.node_weight, cfg.l2_weight)
    train_net = TrainStepWrapCell(loss_net, cfg.lr)
    train_net.set_train()
    eval_net = EvalNet(network)
    time_callback = TimeMonitor(data_size=train_ds.get_dataset_size())
    loss_callback = MyLossMonitor()
    model = Model(network=train_net, eval_network=eval_net, metrics={'auc': AUC(), 'acc': ACC()})
    print("training...")
    for i in range(args.epochs):
        print(f'===================== Epoch {i} =====================')
        model.train(epoch=1, train_dataset=train_ds, callbacks=[time_callback, loss_callback], dataset_sink_mode=False)
        train_out = model.eval(train_ds, dataset_sink_mode=False)
        test_out = model.eval(test_ds, dataset_sink_mode=False)
        print(f'Train AUC:{train_out["auc"]} ACC:{train_out["acc"]}  Test AUC:{test_out["auc"]} ACC:{test_out["acc"]}')

        ckpt_path = os.path.join(ckpt_dir_path, f'tbnet_epoch{i}.ckpt')
        save_checkpoint(network, ckpt_path)
        print(f'checkpoint saved: {ckpt_path}')


if __name__ == '__main__':
    train_tbnet()
