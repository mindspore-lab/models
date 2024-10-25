# Copyright 2022 Huawei Technologies Co., Ltd
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

"""
-*- coding: utf-8 -*-
@File  : dataset.py
"""

import numpy as np
from utils import load_json, load_pickle


class MyDataset:
    def __init__(self, data_path, set_name='train', max_hist_len=10, aug_prefix=None):
        self.max_hist_len = max_hist_len
        self.set = set
        self.data = load_pickle(data_path + f'/ctr.{set_name}')
        self.stat = load_json(data_path + f'/{aug_prefix}_stat.json')
        self.item_num = self.stat['item_num']
        self.attr_num = self.stat['attribute_num']
        self.attr_ft_num = self.stat['attribute_ft_num']
        self.rating_num = self.stat['rating_num']
        self.dense_dim = self.stat['dense_dim']
        self.length = len(self.data)
        self.sequential_data = load_json(data_path + '/sequential_data.json')
        self.item2attribution = load_json(data_path + '/item2attributes.json')
        datamaps = load_json(data_path + '/datamaps.json')
        self.id2item = datamaps['id2item']
        self.id2user = datamaps['id2user']
        self.hist_aug_data = load_json(data_path + f'/{aug_prefix}_augment.hist')
        self.item_aug_data = load_json(data_path + f'/{aug_prefix}_augment.item')
        self.aug_num = 2

    def __len__(self):
        return self.length

    def __getitem__(self, _id):
        uid, seq_idx, label = self.data[_id]
        item_seq, rating_seq = self.sequential_data[str(uid)]
        iid = item_seq[seq_idx]
        hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
        attri_id = self.item2attribution[str(iid)]
        hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
        hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
        hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
        item_aug_vec = self.item_aug_data[str(self.id2item[str(iid)])]
        hist_aug_vec = self.hist_aug_data[str(self.id2user[str(uid)])]

        return iid, attri_id, hist_item_seq, hist_attri_seq, hist_rating_seq,\
               hist_seq_len, np.array(item_aug_vec, dtype=np.float32),\
               np.array(hist_aug_vec, dtype=np.float32), np.array(label, dtype=np.float32)
