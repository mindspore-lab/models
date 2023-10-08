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
"""MCD inference."""
import argparse
import ast
import os
import timeit

import mindspore as ms
from mindspore import set_context
from mindspore.nn import Accuracy
from mindspore.nn import NLLLoss
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from src.dataset import create_svhn2mnist_dataset
from src.svhn2mnist import Net
from src.customfunc import WithEvalCell

def parse_args():
    """Get parameters from command line."""
    parser = argparse.ArgumentParser(description="MCD evaluating.")

    parser.add_argument("--data_url", type=str, default='./MindRecord/')
    parser.add_argument("--train_url", type=str, default=None)
    parser.add_argument("--checkpoint_url", type=str, default=None)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False)

    return parser.parse_args()


def inference(net, dataset):
    acc1 = Accuracy('classification')
    acc2 = Accuracy('classification')
    acc_ensemble = Accuracy('classification')
    acc1.clear()
    acc2.clear()
    acc_ensemble.clear()
    count = 0
    for data in dataset.create_dict_iterator():
        print("=====> Batch: ", count, flush=True)
        img = data['T']
        label = data['T_label']
        pred1, pred2, pred_ensemble, _ = net(img, label)
        acc1.update(pred1, label)
        acc2.update(pred2, label)
        acc_ensemble.update(pred_ensemble, label)
        count += 1
    return acc1.eval(), acc2.eval(), acc_ensemble.eval()


def main():
    """Inference process."""
    args = parse_args()
    if args.modelarts:
        import moxing as mox
        local_data_url = "/cache/data"
        mox.file.copy_parallel(args.data_url, local_data_url)
        ckpt_name = args.checkpoint_url.strip().split("/")[-1]
        local_checkpoint_url = os.path.join("/cache/ckpt", ckpt_name)
        mox.file.copy_parallel(args.checkpoint_url, local_checkpoint_url)
    else:
        local_data_url = args.data_url
        local_checkpoint_url = args.checkpoint_url

    set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

    # Prepare dataset
    _, dataset = create_svhn2mnist_dataset(data_url=local_data_url, num_workers=args.workers)
    step_per_epoch = dataset.get_dataset_size()
    batch_size = dataset.get_batch_size()

    # Initialize network
    net = Net()
    param_dict = load_checkpoint(ckpt_file_name=local_checkpoint_url)
    load_param_into_net(net, param_dict)
    nll_loss = NLLLoss()
    eval_net = WithEvalCell(net, nll_loss)
    eval_net.set_train(False)

    # Calculate results
    start = timeit.default_timer()
    acc1, acc2, acc_ensemble = inference(eval_net, dataset)
    end = timeit.default_timer()
    total_time = end - start

    print(f'Number of samples: {step_per_epoch * batch_size:4d}, total time: {total_time:4.2f} ms', flush=True)

    # Show results
    print("============= 910 Inference =============", flush=True)
    print(f"acc1: {acc1:.4f} | acc2: {acc2:.4f} | acc_ensemble: {acc_ensemble:.4f}",
          flush=True)
    print("=========================================", flush=True)


if __name__ == '__main__':
    main()
