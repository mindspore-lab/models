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


import mindspore.dataset as ds
import pandas as pd
import json
import numpy as np
from models import *


def get_list(df, keys):
    if type(df[keys][0]) != str:
        df[keys] = df[keys].map(lambda x: int(x))
    else:
        df[keys] = df[keys].map(lambda x: json.loads(x))
    input_ids = []
    for i in df[keys]:
        input_ids.append(i)
    return input_ids


def get_list_1(df, keys):
    if type(df[keys][0]) != str:
        df[keys] = df[keys].map(lambda x: int(x))
    else:
        df[keys] = df[keys].map(lambda x: json.loads(x))
    input_ids = []
    for i in df[keys]:
        if len(i) <= 100:
            a = [0 for i in range(100 - len(i))]
            i.extend(a)
            input_ids.append(i)
        else:
            input_ids.append(i[0:100])
    return input_ids


def get_dataset(path="./data/ml-1m/data.csv", batch_size=100):
    df = pd.read_csv(path)
    hist_items = get_list(df, "hist_items")
    pos_items = get_list(df, "pos_items")
    neg_items = get_list(df, "neg_items")
    data1 = {"hist_items": hist_items, "pos_items": pos_items, "neg_items": neg_items}

    dataset1 = ds.NumpySlicesDataset(
        data1, column_names=["hist_items", "pos_items", "neg_items"], shuffle=False
    )
    dataset1 = dataset1.batch(batch_size=batch_size)
    return dataset1


def get_test_dataset(path="./data/ml-1m/test.csv", batch_size=1):
    df = pd.read_csv(path)
    hist_items = get_list(df, "hist_items")
    target_items = get_list_1(df, "target_items")
    data1 = {"hist_items": hist_items, "pos_items": target_items}

    dataset1 = ds.NumpySlicesDataset(
        data1, column_names=["hist_items", "pos_items"], shuffle=False
    )
    dataset1 = dataset1.batch(batch_size=batch_size)

    return dataset1


def read_cate(cate_dir):
    item_cate_map = []
    with open(cate_dir, "r") as f:
        for line in f:
            l = line.strip().split(",")
            item_id, cate_id = int(l[0]), int(l[1])
            item_cate_map.append((item_id, cate_id))
    item_cate_map.sort(key=lambda x: x[0])
    item_cate_map = [x[1] for x in item_cate_map]
    item_cate_map = [0] + item_cate_map
    return item_cate_map, max(item_cate_map)


def get_model(num_items, num_cate, item_cate_map, args):
    if args.model == "DQMatchWithDNN":
        model = DQMatchWithDNN(
            n_items=num_items,
            n_cate=num_cate,
            item_cate_map=item_cate_map,
            seq_len=args.seq_len,
            embedding_size=args.embedding_size,
            l2_reg=args.l2,
            dropout=args.dropout,
            k_c=args.K_c,
            k_u=args.K_u,
            k_i=args.K_i,
            token_embedding_size=args.token_embedding_size,
            beta=args.beta,
            batch_norm=False,
            num_codebooks=args.num_codebooks,
        )
    else:
        model = None
    return model


if __name__ == "__main__":
    dataset = get_test_dataset()
    for batch, data in enumerate(dataset.create_dict_iterator()):
        print(max(data["hist_items"][0]))
        if batch == 100:
            break
