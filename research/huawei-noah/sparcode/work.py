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


import argparse
from models import *
from tqdm import tqdm
import numpy as np
import metrics
import faiss
import time
from load_csv_dataset import get_dataset, read_cate, get_model, get_test_dataset


parser = argparse.ArgumentParser()
# DQMatchWithSASRec
parser.add_argument(
    "--model", default="DQMatchWithDNN", type=str, help="GRU4Rec | YoutubeDNN |"
)
parser.add_argument(
    "--model_type", default="two_tower", type=str, help="VQ or two_tower"
)
parser.add_argument("--mode", default="train", type=str, help="train|test")
parser.add_argument("--dataset", default="ml-1m", type=str)
parser.add_argument("--seq_len", default=20, type=int)
parser.add_argument("--batch_size", default=4096, type=int)
parser.add_argument("--inference_batch_size", default=8192, type=int)
parser.add_argument("--embedding_size", default=64, type=int)
parser.add_argument("--token_embedding_size", default=64, type=int)
parser.add_argument("--sparse_bias", default=1000, type=float)
parser.add_argument(
    "--score_bias",
    default=None,
    type=float,
    help="the scores bias for controlling sparsity " "of scores.",
)
parser.add_argument("--l2", default=1e-6, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--topk", default=[1, 5, 10, 20, 50], type=list)
parser.add_argument("--neg_num", default=10, type=int)
parser.add_argument("--patience", default=5, type=int, help="early stop")
parser.add_argument(
    "--monitor", default="recall", type=str, help="[recall,ndcg,hit_ratio]"
)
parser.add_argument("--debug", default=1, type=int, help="1 or 0")
parser.add_argument("--K_c", default=256, type=int, help="")
parser.add_argument("--K_u", default=4, type=int, help="")
parser.add_argument("--K_i", default=4, type=int, help="")
parser.add_argument("--save_model_by_epoch", default=0, type=int, help="")
parser.add_argument("--num_reported", default=1, type=int)
parser.add_argument("--num_codebooks", default=2, type=int)
parser.add_argument("--att_layer", default=3, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--beta", default=1.0, type=float, help="the weight of vq_loss")
parser.add_argument("--token_dist", default=False, type=bool, help="token distribution")
parser.add_argument(
    "--pretrain_epoch", default=0, type=int, help="pretrain models without vq"
)
parser.add_argument(
    "--best_model_pth",
    default=None,
    type=str,
    help="the path of the best trained model.",
)
args = parser.parse_args()
device = "cpu"


def print_result(result, topk):
    for metrics, v in result.items():
        if metrics in ["epoch", "time"]:
            continue
        elif metrics in ["inference_time"]:
            print("{}:{:.6f} s".format(metrics, v))
            continue
        temp_str = ""
        for i, k in enumerate(topk):
            temp_str += "{}@{}:{:.6f}\t".format(metrics, k, v[i])
        print(temp_str)


def inference(model, data_loader):
    result = {
        "hit_ratio": np.zeros(len(args.topk)),
        "ndcg": np.zeros(len(args.topk)),
        "recall": np.zeros(len(args.topk)),
        "precision": np.zeros(len(args.topk)),
    }
    item_embedding = model.get_item_embedding().asnumpy()  # (I,D)
    max_topk = max(args.topk)
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = args.gpu  # args.faiss_gpu
    index_flat = faiss.GpuIndexFlatIP(res, args.embedding_size, flat_config)
    index_flat.add(item_embedding)
    count = 0
    t1 = time.time()
    for data in tqdm(data_loader.create_dict_iterator()):
        hist_items = data["hist_items"]
        target_items = data["pos_items"]
        count += len(hist_items)
        target_items = target_items.asnumpy()
        user_embedding = model.get_user_embedding(hist_items).asnumpy()
        D, I = index_flat.search(user_embedding[0], k=max_topk)
        for i in range(len(user_embedding)):
            r = []
            for t in I[i]:
                if t in target_items[i]:
                    r.append(1)
                else:
                    r.append(0)
            for j, K in enumerate(args.topk):
                result["hit_ratio"][j] += metrics.hit_at_k(r, K)
                result["ndcg"][j] += metrics.ndcg_at_k(r, K, target_items[i])
                result["recall"][j] += metrics.recall_at_k(r, K, len(target_items[i]))
                result["precision"][j] += metrics.precision_at_k(r, K)

    t2 = time.time()
    result["hit_ratio"] /= count
    result["ndcg"] /= count
    result["recall"] /= count
    result["precision"] /= count
    result["inference_time"] = t2 - t1
    print_result(result, args.topk)
    return result


def forward_fn(data):
    loss, vq_loss = model(data)
    return loss, vq_loss


def train_step(data):
    (loss, _), grads = grad_fn(data)
    m_optimizer(grads)
    return loss


def train_loop(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, data in enumerate(tqdm(dataset.create_dict_iterator())):
        loss = train_step(data)
        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f},  step: [{current} : {size}]")


if __name__ == "__main__":
    cate_dir = "./data/ml-1m/cate_item.txt"
    item_cate_map, n_cate = read_cate(cate_dir)
    n_item = 3952
    n_user = 6040
    model = get_model(n_item, n_cate, item_cate_map, args)
    m_optimizer = nn.Adam(params=model.trainable_params(), learning_rate=0.0005)
    grad_fn = mindspore.value_and_grad(
        forward_fn, None, m_optimizer.parameters, has_aux=False
    )
    dataset = get_dataset(batch_size=100)
    dataset_test = get_test_dataset(batch_size=1)
    for i in range(5):
        train_loop(model, dataset)
        inference(model, dataset_test)
