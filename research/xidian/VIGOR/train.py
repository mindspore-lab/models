# Copyright 2024 Xidian University
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
"""VIGOR training."""
import argparse
import ast
import os
import random
import time
import numpy as np

import mindspore as ms
from mindspore import CheckpointConfig, ModelCheckpoint
from mindspore import Model
from mindspore import ParallelMode
from mindspore import TimeMonitor, LossMonitor
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import save_checkpoint
from mindspore import set_auto_parallel_context
from mindspore import set_context
from mindspore.communication import init
from mindspore.nn import Adam

from src.model import VIGOR
from src.loss import IoULoss, OffsetLoss
from src.dataset import get_dataloader
from src.callbacks import RecallEvalCallback
from engine import TrainStep


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
    return


def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description='MindSpore VIGOR Training')

    parser.add_argument("--data_url", type=str, default='/path/data/VIGOR/', help="Storage path of dataset in OBS.")
    parser.add_argument('--same_area', default=False, type=bool, help='same area.')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--max_epoch', default=250, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--begin_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')

    parser.add_argument("--checkpoint_url", type=str, default=None,
                        help="Storage path of checkpoint for pretraining or resuming in OBS.")
    parser.add_argument("--train_url", type=str, default='/path/res/', help="Storage path of training results in OBS.")
    parser.add_argument('--keep_checkpoint_max', default=10, type=int)
    parser.add_argument('--save_checkpoint_epochs', default=10, type=int)

    parser.add_argument('--eval', default=True, type=bool, help='evaluate when training.')
    parser.add_argument('--eval_start', default=1, type=int, help='evaluate when training.')
    parser.add_argument('--interval', default=1, type=int, metavar='N', help='evaluation frequency (default: 10)')

    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Use one card or multiple cards training.")

    return parser.parse_args()


def main():
    set_seed(1)
    args = parse_args()

    if args.modelarts:
        import moxing as mox
        local_data_url = "/cache/dataset"
        mox.file.copy_parallel(args.data_url, local_data_url)
        local_train_url = "/cache/output"
        if args.checkpoint_url:
            if "obs://" in args.checkpoint_url:
                local_checkpoint_url = "/cache/" + args.checkpoint_url.split("/")[-1]
                mox.file.copy_parallel(args.checkpoint_url, local_checkpoint_url)
            else:
                dir_path = os.path.dirname(os.path.abspath(__file__))
                ckpt_name = args.checkpoint_url[2:]
                local_checkpoint_url = os.path.join(dir_path, ckpt_name)
        else:
            local_checkpoint_url = None
    else:
        local_data_url = args.data_url
        local_train_url = args.train_url
        local_checkpoint_url = args.checkpoint_url

    if args.run_distribute:
        init()
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = int(os.getenv("RANK_SIZE"))
        parallel_mode = ParallelMode.DATA_PARALLEL
        set_auto_parallel_context(parallel_mode=parallel_mode,
                                  gradients_mean=True,
                                  device_num=device_num)
    else:
        device_id = 0

    # set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
    set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=device_id)

    train_loader, val_sat_loader, val_grd_loader, \
        len_train, len_val_sat, len_val_grd = get_dataloader(root=local_data_url, 
                                                             batch_size=args.batch_size,
                                                             num_workers=args.workers, 
                                                             same_area=args.same_area)
    steps_per_epoch = train_loader.get_dataset_size()
    print(f'step per epoch: {steps_per_epoch}')

    net = VIGOR()
    if local_checkpoint_url:
        pretrained_dict = load_checkpoint(local_checkpoint_url)
        load_param_into_net(net, pretrained_dict)
    
    loss1 = IoULoss(); loss2 = OffsetLoss()
    trainStep = TrainStep(net, loss1, loss2)
    optimizer = Adam(net.trainable_params(), learning_rate=args.lr)
    model = Model(network=trainStep, optimizer=optimizer)
    
    # Callbacks
    cb = [TimeMonitor(), LossMonitor(steps_per_epoch)]
    # Save-checkpoint callback
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * args.save_checkpoint_epochs,
                                   keep_checkpoint_max=args.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix=f"VIGOR",
                              directory=os.path.join(local_train_url, f"card{device_id}"),
                              config=ckpt_config)
    cb.append(ckpt_cb)
    # Eval callbacks
    if args.eval:
        eval_cb = RecallEvalCallback(val_grd_loader, val_sat_loader, net, 
                                     start_epoch=args.eval_start, save_path=local_train_url, 
                                     interval=args.interval, 
                                     query_data_size=len_val_grd, 
                                     ref_data_size=len_val_sat)
        cb.append(eval_cb)

    if args.begin_epoch > args.max_epoch:
        raise ValueError("begin epoch should not be larger than total epoch.")
    train_epoch = args.max_epoch - args.begin_epoch

    start = time.time()

    model.train(train_epoch, train_loader, callbacks=cb, dataset_sink_mode=True)

    print(f'\nTraining finished, consume:{time.time() - start:.4f} ms\n')

    last_checkpoint = os.path.join(local_train_url, f"VIGOR_{device_id}_final.ckpt")
    save_checkpoint(net, last_checkpoint)


if __name__ == '__main__':
    main()
