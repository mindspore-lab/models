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

import os
from src.utils.serialization import json_load
from src.datasets.base_dataset import BaseDataset

DEFAULT_INFO_PATH = os.path.join(os.path.dirname(__file__),'../cityscapes_list/info.json')


class CityscapesDataSet(BaseDataset):
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path=DEFAULT_INFO_PATH, labels_size=None):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)

        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str_)
        self.mapping = np.array(self.info['label2train'], dtype=np.int_)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        img_file = self.root / 'leftImg8bit' / self.set / name
        label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        label_file = self.root / 'gtFine' / self.set / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        label = self.get_labels(label_file)
        label = self.map_labels(label).copy()
        image = self.get_image(img_file)
        image = self.preprocess(image)
        return image.copy().astype(np.float32), label.astype(np.int32),name


if __name__ == '__main__':
    import mindspore.dataset as ds

    test_dataset = CityscapesDataSet(root='./data/Cityscapes',
                                     list_path=f'./dataset/cityscapes_list/train.txt',
                                     set='train',
                                     info_path='./dataset/cityscapes_list/info.json',
                                     crop_size=(1024,512),
                                     mean=np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32),
                                     labels_size=(2048,1024))

    test_loader = ds.GeneratorDataset(test_dataset, ["image", "label","name"], shuffle=False)
    test_loader = test_loader.batch(4)

    for data in test_loader.create_dict_iterator():
        image = data['image']
        label = data['label']
        print(np.unique(label.asnumpy()))