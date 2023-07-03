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
"""Dataset setting and data loader for USPS."""

import gzip
import os
import pickle
import urllib
import numpy as np
import mindspore
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore.dataset.vision import c_transforms, py_transforms
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
from model_utils.config import config


class GetDatasetGenerator:
    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.current_dir, "../data/USPS", self.filename)
        # download dataset.
        if download:
            self.download()

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data *= 255.0
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = Tensor(label, mindspore.int64)
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        f = gzip.open(self.data_path, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels


def get_usps(train):
    """Get USPS dataset loader."""
    pre_process = py_transforms.ToTensor()
    dataset_generator = GetDatasetGenerator(root=config.data_root,
                                            train=train,
                                            transform=pre_process,
                                            download=False)

    dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=True)
    onehot_op = C.OneHot(num_classes=10)
    type_cast_op = C.TypeCast(mstype.float32)
    normalize_op = c_transforms.Normalize(mean=[(config.dataset_mean)], std=[(config.dataset_std)])
    dataset = dataset.map(operations=onehot_op, input_columns="label", num_parallel_workers=1)
    dataset = dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)
    dataset = dataset.map(operations=normalize_op, input_columns="image", num_parallel_workers=1)
    dataset = dataset.batch(batch_size=config.batch_size, drop_remainder=True)

    return dataset
    
