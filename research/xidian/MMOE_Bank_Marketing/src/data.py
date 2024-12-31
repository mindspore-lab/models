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
"""Generate data in mindrecord format."""
import os

import pandas as pd
import numpy as np

from mindspore.mindrecord import FileWriter

from model_utils.config import config

2
def generate_npy(data_path, do_train):
    """create npy file"""
    column_names = ['age', 'job', 'marital_stat','education', 'default', 'housing', 'loan',
                    'concat', 'month', 'day_of_week ', 'duration ', 'campaign ', 'pdays ',
                    'previous ', 'poutcome ', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                    'euribor3m', 'nr.employed','income_50k']
    label_columns = ['income_50k', 'marital_stat']
    categorical_columns = ['job','education', 'default', 'housing', 'loan',
                           'concat', 'month', 'day_of_week ',
                           'poutcome ']
    if do_train:
        ds = pd.read_csv(
            data_path + '/bank-marketing.data.gz',
            delimiter=',',
            index_col=None,
            names=column_names
        )
    else:
        ds = pd.read_csv(
            data_path + '/bank-marketing.test.gz',
            delimiter=',',
            index_col=None,
            names=column_names
        )
    ds_transformed = pd.get_dummies(
        ds.drop(label_columns, axis=1), columns=categorical_columns)
    if not do_train:
        ds_transformed['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0
    data = ds_transformed
    np.save(data_path + '/data.npy', np.array(data), allow_pickle=False)

    ds_raw_labels = ds[label_columns]
    ds_raw_labels['marital_stat'] = ds_raw_labels['marital_stat'].apply(
        lambda x: 'married' if x == 'married' else 'unmarried')

    income_labels = pd.get_dummies(ds_raw_labels['income_50k'])
    np.save(data_path + '/income_labels.npy',
            np.array(income_labels), allow_pickle=False)

    married_labels = pd.get_dummies(ds_raw_labels['marital_stat'])
    np.save(data_path + '/married_labels.npy',
            np.array(married_labels), allow_pickle=False)

    data = np.load(data_path + '/data.npy').astype(np.float32)
    income = np.load(data_path + '/income_labels.npy').astype(np.float32)
    married = np.load(data_path + '/married_labels.npy').astype(np.float32)

    mindrecord_path = data_path

    if not os.path.exists(mindrecord_path):
        os.mkdir(mindrecord_path)

    if do_train:
        MINDRECORD_FILE = mindrecord_path + "/train.mindrecord"
        writer = FileWriter(file_name=MINDRECORD_FILE, shard_num=1)

        nlp_schema = {"data": {"type": "float32", "shape": [-1]},
                      "income_labels": {"type": "float32", "shape": [-1]},
                      "married_labels": {"type": "float32", "shape": [-1]}}
        writer.add_schema(nlp_schema, "it is a preprocessed nlp dataset")
        for i in range(len(data)):
            sample = {"data": data[i],
                      "income_labels": income[i],
                      "married_labels": married[i]}

            if i % 10000 == 0:
                print(f'write {i} lines.')

            writer.write_raw_data([sample])
        writer.commit()
    else:
        MINDRECORD_FILE = mindrecord_path + "/eval.mindrecord"
        writer = FileWriter(file_name=MINDRECORD_FILE, shard_num=1)

        nlp_schema = {"data": {"type": "float32", "shape": [-1]},
                      "income_labels": {"type": "float32", "shape": [-1]},
                      "married_labels": {"type": "float32", "shape": [-1]}}
        writer.add_schema(nlp_schema, "it is a preprocessed nlp dataset")
        for i in range(len(data)):
            sample = {"data": data[i],
                      "income_labels": income[i],
                      "married_labels": married[i]}

            if i % 10000 == 0:
                print(f'write {i} lines.')

            writer.write_raw_data([sample])
        writer.commit()


if __name__ == '__main__':
    generate_npy(data_path=config.local_data_path, do_train=True)
    generate_npy(data_path=config.local_data_path, do_train=False)
