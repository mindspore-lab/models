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
"""MCD training on modelarts."""
import argparse
import ast
import os
import random
import numpy as np

import mindspore as ms
from mindspore import CheckpointConfig, ModelCheckpoint
from mindspore import Model
from mindspore import ParallelMode
from mindspore import Tensor
from mindspore import TimeMonitor, LossMonitor
from mindspore import export
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import save_checkpoint
from mindspore import set_auto_parallel_context
from mindspore import set_context
from mindspore.communication import init
from mindspore.nn import Accuracy
from mindspore.nn import Adam
from mindspore.nn import NLLLoss
from mindspore.nn import SoftmaxCrossEntropyWithLogits

from src.callback import EvalCallBack
from src.customfunc import WithEvalCell
from src.customfunc import DiscrepancyLoss
from src.customfunc import StepAWithLossCell, StepBWithLossCell, StepCWithLossCell
from src.customfunc import TrainStep
from src.customfunc import TrainStepACell, TrainStepBCell, TrainStepCCell
from src.dataset import create_svhn2mnist_dataset
from src.svhn2mnist import Net


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
    return


def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description='MindSpore matchNet Training')

    parser.add_argument("--data_url", type=str, default='./MindRecord/', help="Storage path of dataset in OBS.")
    parser.add_argument('--workers', default=64, type=int, metavar='N',
                        help='number of data loading workers (default: 64)')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='mini-batch size (default: 64), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--max_epoch', default=1000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--begin_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, metavar='LR',
                        help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--num_k', type=int, default=4, metavar='N',
                        help='hyper parameter for generator update')

    parser.add_argument("--checkpoint_url", type=str, default=None,
                        help="Storage path of checkpoint for pretraining or resuming in OBS.")
    parser.add_argument("--train_url", type=str, default='./res/', help="Storage path of training results in OBS.")
    parser.add_argument('--keep_checkpoint_max', default=10, type=int)
    parser.add_argument('--save_checkpoint_epochs', default=10, type=int)

    parser.add_argument('--eval', default=True, type=bool, help='evaluate when training.')
    parser.add_argument('--eval_per_epoch', default=1, type=int, metavar='N', help='evaluation frequency (default: 10)')

    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Use one card or multiple cards training.")
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False,
                        help="Run on ModelArts or offline machines.")

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
    train_set, test_set = create_svhn2mnist_dataset(args.batch_size, local_data_url, args.workers)
    steps_per_epoch = train_set.get_dataset_size()

    # Create network
    net = Net()
    if local_checkpoint_url:
        pretrained_dict = load_checkpoint(local_checkpoint_url)
        load_param_into_net(net, pretrained_dict)

    # Create loss
    ce_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    discrepancy = DiscrepancyLoss()
    nll_loss = NLLLoss()

    # Create optimizer
    optimizer_G = Adam(net.G.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)
    optimizer_C1 = Adam(net.C1.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)
    optimizer_C2 = Adam(net.C2.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)

    # Connect net with loss
    stepAWithLoss = StepAWithLossCell(net, ce_loss)
    stepBWithLoss = StepBWithLossCell(net, ce_loss, discrepancy)
    stepCWithLoss = StepCWithLossCell(net, discrepancy)

    # Connect with optimizer
    stepA = TrainStepACell(stepAWithLoss, optimizer_G, optimizer_C1, optimizer_C2)
    stepB = TrainStepBCell(stepBWithLoss, optimizer_C1, optimizer_C2)
    stepC = TrainStepCCell(stepCWithLoss, optimizer_G)
    num_k = ms.Tensor(args.num_k)
    train_step = TrainStep(stepA, stepB, stepC, num_k)
    train_step.set_train(True)

    # Create model
    model = Model(network=train_step, boost_level='O1')

    # Callbacks
    cb = [TimeMonitor(), LossMonitor(steps_per_epoch)]
    # Save-checkpoint callback
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * args.save_checkpoint_epochs,
                                   keep_checkpoint_max=args.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix=f"MCD",
                              directory=os.path.join(local_train_url, f"card{device_id}"),
                              config=ckpt_config)
    cb.append(ckpt_cb)

    if args.eval:
        eval_step = WithEvalCell(network=net, loss_fn=nll_loss)
        eval_step.set_train(False)
        # Metric
        acc1 = Accuracy('classification')
        acc2 = Accuracy('classification')
        acc_ensemble = Accuracy('classification')
        metrics = {'acc1': acc1,
                   'acc2': acc2,
                   'acc_ensemble': acc_ensemble}
        eval_cb = EvalCallBack(eval_step, test_set, metrics, args.eval_per_epoch)
        cb.append(eval_cb)

    if args.begin_epoch > args.max_epoch:
        raise ValueError("begin epoch should not be larger than total epoch.")
    train_epoch = args.max_epoch - args.begin_epoch

    model.train(epoch=train_epoch, train_dataset=train_set, callbacks=cb, dataset_sink_mode=True)

    last_checkpoint = os.path.join(local_train_url, f"MCD-card{device_id}-final.ckpt")
    save_checkpoint(net, last_checkpoint)

    if device_id == 0:
        ckpt_model = last_checkpoint
        print("Checkpoint path: ", ckpt_model)
        net = Net()
        param_dict = load_checkpoint(ckpt_file_name=ckpt_model)
        load_param_into_net(net, param_dict, strict_load=True)
        net.set_train(False)
        file_name = os.path.join(local_train_url, f"MCD_{args.num_k}")
        input_data = Tensor(np.zeros([1, 3, 32, 32], dtype=np.float32))
        export(net, input_data, file_name=file_name, file_format=args.file_format)
        print("Export AIR model successfully.", flush=True)

    if args.modelarts:
        import moxing as mox
        mox.file.copy_parallel(local_train_url, args.train_url)


if __name__ == "__main__":
    main()
