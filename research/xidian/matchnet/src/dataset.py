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
import numpy as np

from mindspore.dataset import MindDataset


def PreProcessing(vector):
    """
    vector:ndarray, shape of (1, weight*height),
            dtype=uint8
    """
    img = vector.reshape((64, 64)).astype('float32')
    img = (img - 128.0 ) / 160.0
    return np.expand_dims(img, axis=0)


def DataLoader(batch_size=1024, dataset_dir='./MindRecord/', name='liberty', training=True, num_workers=16, shuffle=False):
    if training:
        usage = 'train'
    else:
        usage = 'test'
    mind_dataset_dir = dataset_dir + '/' + name + '/' + name + '-' + usage + '.mindrecord'
    dataloader = MindDataset(dataset_files=mind_dataset_dir, num_parallel_workers=num_workers, shuffle=shuffle)
    dataloader = dataloader.map(operations=PreProcessing, input_columns=["image1"])
    dataloader = dataloader.map(operations=PreProcessing, input_columns=["image2"])
    dataloader = dataloader.batch(batch_size)
    return dataloader
