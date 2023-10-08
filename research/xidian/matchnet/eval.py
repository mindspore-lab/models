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
"""MatchNet inference."""
import os
import ast
import timeit
import argparse
from scipy import interpolate

import mindspore as ms
from mindspore import set_context
from mindspore.nn import ROC
from mindspore.nn import Softmax
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from src.MatchNet import MatchNet
from src.customfunc import WithEvalCell
from src.dataset import DataLoader


def parse_args():
    """Get parameters from command line."""
    parser = argparse.ArgumentParser(description="MatchNet evaluating.")

    parser.add_argument("--data_url", type=str, default='./MindRecord/')
    parser.add_argument("--dataset", type=str, default="liberty")
    parser.add_argument("--train_url", type=str, default=None)
    parser.add_argument("--checkpoint_url", type=str, default=None)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False)

    return parser.parse_args()


def inference(eval_net, dataset):
    roc = ROC(class_num=1)
    roc.clear()
    count = 0
    for data in dataset.create_dict_iterator():
        print("=====> Batch: ", count, flush=True)
        pred_scores = eval_net(data["image1"], data["image2"])
        roc.update(pred_scores, data["matches"])
        count += 1
    fpr, tpr, thresholds = roc.eval()
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
    return fpr95


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
    dataset = DataLoader(dataset_dir=local_data_url, name=args.dataset,
                         num_workers=args.workers, training=False, shuffle=False)
    step_per_epoch = dataset.get_dataset_size()
    batch_size = dataset.get_batch_size()

    print(args.dataset, step_per_epoch, batch_size)

    # Initialize network
    net = MatchNet()
    param_dict = load_checkpoint(ckpt_file_name=local_checkpoint_url)
    load_param_into_net(net, param_dict)
    eval_net = WithEvalCell(net)
    eval_net.set_train(False)

    # Calculate results
    start = timeit.default_timer()
    fpr95 = inference(eval_net, dataset)
    end = timeit.default_timer()
    total_time = end - start

    print(f'Number of samples: {step_per_epoch * batch_size:4d}, total time: {total_time:4.2f} ms', flush=True)

    # Show results
    print("============= 910 Inference =============", flush=True)
    print(f"FPR@95: {fpr95:.6f}", flush=True)
    print("=========================================", flush=True)


if __name__ == '__main__':
    main()
