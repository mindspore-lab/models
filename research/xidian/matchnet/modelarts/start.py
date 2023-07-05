# Copyright 2023 Xidian University
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
"""MatchNet training on modelarts."""
import os
import ast
import random
import argparse
import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore import set_context
from mindspore import Model
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import SGD
from mindspore.nn import ROC
from mindspore import TimeMonitor, LossMonitor
from mindspore import CheckpointConfig, ModelCheckpoint
from mindspore import save_checkpoint
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import export
from mindspore import ParallelMode
from mindspore import set_auto_parallel_context
from mindspore.communication import init

from src.MatchNet import MatchNet
from src.customfunc import WithLossCell, WithEvalCell
from src.callback import EvalCallBack
from src.dataset import DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
    return


def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description='MindSpore matchNet Training')

    parser.add_argument("--data_url", type=str, default='./MindRecord/', help="Storage path of dataset in OBS.")
    parser.add_argument('--dataset', default='notredame', type=str, help='liberty, notredame or yosemite')
    parser.add_argument('--workers', default=64, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--max_epoch', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--begin_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')

    parser.add_argument("--checkpoint_url", type=str, default=None,
                        help="Storage path of checkpoint for pretraining or resuming in OBS.")
    parser.add_argument("--train_url", type=str, default='./res/', help="Storage path of training results in OBS.")
    parser.add_argument('--keep_checkpoint_max', default=10, type=int)
    parser.add_argument('--save_checkpoint_epochs', default=10, type=int)

    parser.add_argument('--eval', default=True, type=bool, help='evaluate when training.')
    parser.add_argument('--evalset', default='liberty', type=str, help='liberty, notredame or yosemite')
    parser.add_argument('--eval_per_epoch', default=1, type=int, metavar='N', help='print frequency (default: 10)')

    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Use one card or multiple cards training.")
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False,
                        help="Run on ModelArts or offline machines.")
    parser.add_argument("--file_format", type=str, default="MINDIR",
                        choices=["AIR", "MINDIR"], help="Output file format. ")

    return parser.parse_args()


def main():
    """Training process."""
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

    set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

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

    # Create dataset
    train_dataset = DataLoader(batch_size=args.batch_size, dataset_dir=local_data_url, name=args.dataset,
                               num_workers=args.workers, shuffle=True)
    steps_per_epoch = train_dataset.get_dataset_size()

    # Create network
    net = MatchNet()
    if local_checkpoint_url:
        pretrained_dict = load_checkpoint(local_checkpoint_url)
        load_param_into_net(net, pretrained_dict)
    net.set_train(True)

    # Create loss
    criterion = SoftmaxCrossEntropyWithLogits(sparse=True)
    net_with_loss = WithLossCell(net, criterion)

    # Create optimizer
    optimizer = SGD(net.trainable_params(), learning_rate=args.lr,
                    weight_decay=args.weight_decay, momentum=args.momentum)

    # Create model
    model = Model(net_with_loss, optimizer=optimizer)

    # Callbacks
    cb = []
    cb.append(TimeMonitor())
    cb.append(LossMonitor(steps_per_epoch))
    # Save-checkpoint callback
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * args.save_checkpoint_epochs,
                                   keep_checkpoint_max=args.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix=f"MatchNet-",
                              directory=os.path.join(local_train_url, f"card{device_id}"),
                              config=ckpt_config)
    cb.append(ckpt_cb)

    if args.eval:
        test_dataset = DataLoader(dataset_dir=local_data_url, name=args.evalset, num_workers=args.workers, shuffle=False)
        eval_net = WithEvalCell(net)
        eval_net.set_train(False)
        roc = ROC(class_num=1)
        eval_cb = EvalCallBack(eval_net, test_dataset, roc, args.eval_per_epoch)
        cb.append(eval_cb)

    if args.begin_epoch > args.max_epoch:
        raise ValueError("begin epoch should not be larger than total epoch.")
    train_epoch = args.max_epoch - args.begin_epoch

    model.train(epoch=train_epoch, train_dataset=train_dataset, callbacks=cb, dataset_sink_mode=True)

    last_checkpoint = os.path.join(local_train_url,
                                   f"matchnet-{args.train_set1}-{args.train_set2}-{device_id}-final.ckpt")
    save_checkpoint(net, last_checkpoint)

    if device_id == 0:
        ckpt_model = last_checkpoint
        print("Checkpoint path: ", ckpt_model)
        net = MatchNet()
        param_dict = load_checkpoint(ckpt_file_name=ckpt_model)
        load_param_into_net(net, param_dict, strict_load=True)
        net.set_train(False)
        file_name = os.path.join(local_train_url, f"MatchNet_{args.dataset}")
        input_data = Tensor(np.zeros([1, 1, 64, 64], dtype=np.float32))
        export(net, input_data, file_name=file_name, file_format=args.file_format)
        print("Export AIR model successfully.", flush=True)

    if args.modelarts:
        import moxing as mox
        mox.file.copy_parallel(local_train_url, args.train_url)


if __name__ == "__main__":
    main()
