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
import os

import mindspore.dataset as dataset
from mindspore.dataset import transforms
from mindspore.dataset import vision


def create_svhn2mnist_dataset(batch_size=256, data_url='./MindRecord', num_workers=16, shuffle=False):
    def reshape(img):
        return img.reshape(32, 32, 3)

    transform = transforms.Compose([
        reshape,
        vision.ToTensor(),
        vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False),
    ])

    train_loader1 = dataset.MindDataset(
        dataset_files=os.path.join(data_url, 'source_train.mindrecord'),
        columns_list=['S', 'S_label'],
        shuffle=shuffle,
        num_parallel_workers=num_workers)
    train_loader1 = train_loader1.map(operations=transform, input_columns=['S'])
    train_loader2 = dataset.MindDataset(
        dataset_files=os.path.join(data_url, 'target_train.mindrecord'),
        columns_list=['T', 'T_label'],
        shuffle=shuffle,
        num_parallel_workers=num_workers)
    train_loader2 = train_loader2.map(operations=transform, input_columns=['T'])

    train_loader1 = train_loader1.zip(train_loader2)
    train_loader1 = train_loader1.batch(batch_size, drop_remainder=True)

    test_loader1 = dataset.MindDataset(
        dataset_files=os.path.join(data_url, 'source_test.mindrecord'),
        columns_list=['S', 'S_label'],
        shuffle=False,
        num_parallel_workers=num_workers)
    test_loader1 = test_loader1.map(operations=transform, input_columns=['S'])
    test_loader2 = dataset.MindDataset(
        dataset_files=os.path.join(data_url, 'target_test.mindrecord'),
        columns_list=['T', 'T_label'],
        shuffle=False,
        num_parallel_workers=num_workers)
    test_loader2 = test_loader2.map(operations=transform, input_columns=['T'])

    test_loader1 = test_loader1.zip(test_loader2)
    test_loader1 = test_loader1.batch(batch_size, drop_remainder=True)

    return train_loader1, test_loader1
