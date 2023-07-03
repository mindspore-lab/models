# Copyright 2020 Huawei Technologies Co., Ltd
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
import os.path as osp
import numpy as np
from random import shuffle
import mindspore.dataset as ds

from PIL import Image, ImageFile


class cityscapesDataSet():
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))  # 计算最大的迭代次数
            self.img_ids = self.img_ids[:max_iters]

        self.files = []
        self.set = set

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            self.files.append({
                "img": img_file,
                "name": name
            })
        if set == 'train':
            shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size)

        image = np.asarray(image, np.float32)
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        return image.copy(), name


if __name__ == '__main__':
    dataset_generator = cityscapesDataSet(r"E:\data\Cityscapes\leftImg8bit_trainvaltest",
                                          './cityscapes_list/val.txt',
                                          crop_size=(1024, 512))
    dataset = ds.GeneratorDataset(dataset_generator, column_names=['image', 'size'], shuffle=False)
    dataset = dataset.batch(batch_size=2)

    dataset_iterator = dataset.create_tuple_iterator()
    for i in range(10):
        data = next(dataset_iterator)
        # print(data)
        img, size = data
        print(img.shape)
        print(dataset_generator.files[i]['name'])
