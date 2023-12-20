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


def get_list(df,keys):
    if type(df[keys][0]) !=str:
        df[keys] = df[keys].map(lambda x: int(x))
    else:
        df[keys] = df[keys].map(lambda x: json.loads(x))
    input_ids = []
    for i in df[keys]:
        input_ids.append(i)
    return input_ids



def get_local_global_maps():
    df = pd.read_csv('./data/ListContUni/aotm-n10/local_global_maps.csv')
    local_global_maps = get_list(df, 'local_global_maps')
    return local_global_maps


def get_dataset_all(path='./train_data.csv',batch_size=100):
    df = pd.read_csv(path)
    input_ids = get_list(df,'input_ids')
    index = get_list(df,'index')
    attention_mask = get_list(df,'attention_mask')
    segment_ids = get_list(df,'segment_ids')
    k_cluster = get_list(df,'k_cluster')
    p_cluster = get_list(df,'p_cluster')
    __special_id = get_list(df,'__special_id')
    k_global = get_list(df,'k_global')
    k_local = get_list(df,'k_local')
    p_global = get_list(df,'p_global')
    p_local = get_list(df,'p_local')
    mask_labels = get_list(df,'mask_labels')
    mask_labels_col_k_cluster = get_list(df,'mask_labels_col_k_cluster')
    mask_labels_col_p_cluster = get_list(df,'mask_labels_col_p_cluster')
    mask_labels_col___special_id = get_list(df,'mask_labels_col___special_id')
    data1 = {'input_ids': input_ids, 'index':index,'attention_mask': attention_mask, 'segment_ids': segment_ids,
                'k_cluster': k_cluster, 'p_cluster': p_cluster, '__special_id': __special_id, 'k_global': k_global,
                'k_local': k_local, 'p_global': p_global, 'p_local': p_local,'mask_labels':mask_labels,
                'mask_labels_col_k_cluster':mask_labels_col_k_cluster,
                'mask_labels_col_p_cluster':mask_labels_col_p_cluster,'mask_labels_col___special_id':mask_labels_col___special_id}


    dataset1 = ds.NumpySlicesDataset(data1, column_names=["input_ids", "index","attention_mask", "segment_ids",
                                                         "k_cluster", "p_cluster","__special_id", "k_global","k_local",
                                                         "p_global","p_local","mask_labels","mask_labels_col_k_cluster","mask_labels_col_p_cluster"
                                                          ,"mask_labels_col___special_id"], shuffle=False)
    dataset1 = dataset1.batch(batch_size=batch_size)

    return dataset1

class Batch(object):
    def __init__(self, batch):
        self.input_ids = batch['input_ids']  # type:
        self.attention_mask = batch['attention_mask']  # type:
        self.segment_ids = batch['segment_ids']  #
        self.col_mask = {'k_cluster':batch['k_cluster'],
                         'p_cluster': batch['p_cluster'],
                         '__special_id': batch['__special_id']}  # type:
        self.attr_ids = {'k_cluster': {'k_global': batch['k_global'],'k_local': batch['k_local']},
                         'p_cluster': {'p_global': batch['p_global'],'p_local': batch['p_local']}}
        self.batch_size = int(self.input_ids.shape[0])
        self.seq_size = int(self.input_ids.shape[1])
        self.mask_labels_col = {'k_cluster': batch['mask_labels_col_k_cluster'],
                         'p_cluster': batch['mask_labels_col_p_cluster'],
                         '__special_id': batch['mask_labels_col___special_id']}
        self.mask_labels = batch['mask_labels']
        self.append_info = batch['index']
        self.weight = 1
        self.mask_ratio = 0.2
        self.register = {'segment_ids', 'mask_labels_col', 'append_info', 'attention_mask', 'input_ids', 'col_mask', 'weight', 'mask_ratio', 'attr_ids', 'mask_labels', 'task'}


if __name__ == '__main__':
    dataset = get_dataset_all(path='./test_all.csv',batch_size=1)
    for i in dataset.create_dict_iterator():
        print(i['index'])
        batch = Batch(i)
        print(batch.input_ids)
        # break
