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

import mindspore
from PIL import Image
import os
import mindspore.dataset as ds
import numpy as np
import  mindspore as ms


class GetLoader():
    def __init__(self, data_root, data_list, transform=None):
        # super(GetLoader, self).__init__()
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)
        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        imgs = np.array(imgs)
        labels = np.array(int(labels))


        return imgs, labels

    def __len__(self):
        return self.n_data

import mindspore.dataset.vision.c_transforms as c_vision

image_size= 28
img_transform_target_list =[
    c_vision.RandomResize(28),
    c_vision.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    c_vision.HWC2CHW()
]

