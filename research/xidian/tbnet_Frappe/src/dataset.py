# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Path dataset loader."""

from functools import partial

import numpy as np
from mindspore.dataset import GeneratorDataset


def create_dataset(data_path, per_item_paths, **kwargs):
    """
    Create a dataset for TB-Net.

    Note:
        Columns: 'items', 'relation1', 'reference', 'relation2', 'hist_item', 'label'.

    Args:
        data_path (str): The csv datafile path.
        per_item_paths (int): The number of paths per item.
        **kwargs (any): Other arguments for GeneratorDataset(), except 'source' and 'column_names'.

    Returns:
        GeneratorDataset, the generator dataset that reads from the csv datafile.

    Raises:
        IOError: Be raised for any file content problem.
    """

    kwargs['source'] = partial(_csv_generator, data_path, per_item_paths)
    kwargs['column_names'] = ['item', 'relation1', 'reference', 'relation2', 'hist_item', 'label']
    return GeneratorDataset(**kwargs)


def _csv_generator(csv_path, per_item_paths):
    """Generator for csv datafile."""
    expected_columns = 2 + (per_item_paths * 4)
    file = open(csv_path)
    for line in file:
        line = line.strip()
        if not line or line[0] == '#':
            continue
        id_list = line.split(',')
        if len(id_list) < expected_columns:
            raise IOError(f'Expecting {expected_columns} values but got {len(id_list)} only!')
        id_list = list(map(int, id_list))
        item = id_list[0]
        label = id_list[1]

        relation1 = np.empty(shape=(per_item_paths,), dtype=int)
        reference = np.empty_like(relation1)
        relation2 = np.empty_like(relation1)
        hist_item = np.empty_like(relation1)

        for p in range(per_item_paths):
            offset = 2 + (p * 4)
            relation1[p] = id_list[offset]
            reference[p] = id_list[offset + 1]
            relation2[p] = id_list[offset + 2]
            hist_item[p] = id_list[offset + 3]

        # item, relation1, reference, relation2, hist_item, label
        yield np.array(item, dtype=np.int32), relation1, reference, relation2, hist_item, \
            np.array(label, dtype=np.float32)
