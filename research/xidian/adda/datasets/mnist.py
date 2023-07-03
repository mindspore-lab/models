# Copyright 2021 Xidian University
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
"""Dataset setting and data loader for MNIST."""

import os
from mindspore.dataset.vision import c_transforms
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset import MnistDataset
from mindspore import dtype as mstype
from model_utils.config import config


def get_mnist(usage='train', batch_size=config.batch_size,
              num_parallel_workers=1):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/MNIST")
    mnist_ds = MnistDataset(dataset_dir=data_path, usage=usage)
    rescale = 2.0 / 255.0
    shift = -1.0
    normalize_op = c_transforms.Normalize(mean=[(config.dataset_mean)], std=[(config.dataset_std)])
    rescale_op = c_transforms.Rescale(rescale, shift)
    hwc2chw_op = c_transforms.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.float32)
    onehot_op = C.OneHot(num_classes=10)
    mnist_ds = mnist_ds.map(operations=onehot_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=normalize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    return mnist_ds
