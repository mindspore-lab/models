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
"""VIGOR inference."""
import argparse
import ast
import os
import time
import timeit
import numpy as np

import mindspore as ms
from mindspore import set_context
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from src.model import VIGOR
from src.dataset import get_dataloader

def parse_args():
    """Get parameters from command line."""
    parser = argparse.ArgumentParser(description="VIGOR evaluating.")

    parser.add_argument("--data_url", type=str, default='/path/data/VIGOR')
    parser.add_argument("--train_url", type=str, default=None)
    parser.add_argument("--checkpoint_url", type=str, default=None)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--same_area", type=bool, default=False)
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False)

    return parser.parse_args()


def inference(model, query_loader, ref_loader, query_data_size, ref_data_size):
    query_features, reference_features, query_labels = get_descriptor(model, query_loader, ref_loader, query_data_size, ref_data_size)
    RAt1, RAt5, RAt10 = validate(query_features, reference_features, query_labels)
    return RAt1, RAt5, RAt10

def get_descriptor(model, query_loader, ref_loader, query_data_size, ref_data_size):
    query_features = np.zeros([query_data_size, 4096])
    query_labels = np.zeros([query_data_size], dtype=np.int32)
    reference_features = np.zeros([ref_data_size, 4096])
    for batch in query_loader:
        image, index, label = batch
        query_embed = model.inference(image, 'query')
        query_features[index.asnumpy(), :] = query_embed.asnumpy()
        query_labels[index.asnumpy()] = label.asnumpy()
    for batch in ref_loader:
        image, index = batch
        reference_embed = model.inference(image, 'ref')
        reference_features[index.asnumpy().astype(int), :] = reference_embed.asnumpy()
    return query_features, reference_features, query_labels

def validate(query_features, reference_features, query_labels, topk=[1, 5, 10]):
    ts = time.time()
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])
    if N < 80000:
        query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
        reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
        similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())

        for i in range(N):
            ranking = np.sum((similarity[i,:]>similarity[i, query_labels[i]])*1.)

            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.
    else:
        # split the queries if the matrix is too large, e.g. VIGOR
        assert N % 4 == 0
        N_4 = N // 4
        for split in range(4):
            query_features_i = query_features[(split*N_4):((split+1)*N_4), :]
            query_labels_i = query_labels[(split*N_4):((split+1)*N_4)]
            query_features_norm = np.sqrt(np.sum(query_features_i ** 2, axis=1, keepdims=True))
            reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
            similarity = np.matmul(query_features_i / query_features_norm,
                                (reference_features / reference_features_norm).transpose())
            for i in range(query_features_i.shape[0]):
                ranking = np.sum((similarity[i, :] > similarity[i, query_labels_i[i]])*1.)
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.
    results = results/ query_features.shape[0] * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
    return results[:3]

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
    train_loader, val_sat_loader, val_grd_loader, \
        len_train, len_val_sat, len_val_grd = get_dataloader(root=local_data_url, 
                                                             batch_size=args.batch_size,
                                                             num_workers=args.workers, 
                                                             same_area=args.same_area)

    # Initialize network
    net = VIGOR()
    param_dict = load_checkpoint(ckpt_file_name=local_checkpoint_url)
    load_param_into_net(net, param_dict)

    # Calculate results
    start = timeit.default_timer()
    RAt1, RAt5, RAt10 = inference(net, val_grd_loader, val_sat_loader, len_val_grd, len_val_sat)
    end = timeit.default_timer()
    total_time = end - start

    print(f'Number of query: {len_val_grd} ref: {len_val_sat:4d}, total time: {total_time:4.2f} ms', flush=True)

    # Show results
    print("============= 910 Inference =============", flush=True)
    print(f"R@1: {RAt1:.4f} | R@5: {RAt5:.4f} | R@10: {RAt10:.4f}",
          flush=True)
    print("=========================================", flush=True)


if __name__ == '__main__':
    main()
