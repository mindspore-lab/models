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
import os.path as osp
import numpy as np
from random import shuffle
import mindspore.dataset as ds

from PIL import Image, ImageFile
import matplotlib.pyplot as plt


def save_label(label, class_color_map=None, save_path=None):
    label_show = Image.fromarray(label.astype('uint8')).convert('P')
    if class_color_map is not None:
        label_show.putpalette(class_color_map)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        label_show.save(save_path)
    else:
        plt.imshow(label_show)
        plt.show()


city_palette = [128, 64, 128,
           244, 35, 232,
           70, 70, 70,
           102, 102, 156,
           190, 153, 153,
           153, 153, 153,
           250, 170, 30,
           220, 220, 0,
           107, 142, 35,
           152, 251, 152,
           70, 130, 180,
           220, 20, 60,
           255, 0, 0,
           0, 0, 142,
           0, 0, 70,
           0, 60, 100,
           0, 80, 100,
           0, 0, 230,
           119, 11, 32]


class cityscapesDataSet():
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))  # 计算最大的迭代次数
            self.img_ids = self.img_ids[:max_iters]

        self.files = []
        self.set = set

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name.replace("leftImg8bit", "gtFine_labelTrainIds")))
            self.files.append({
                "img": img_file,
                'label': label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = os.path.basename(datafiles["name"])

        # resize
        image = image.resize(self.crop_size)
        # label = label.resize(self.crop_size, resample=Image.NEAREST)

        label = np.asarray(label,np.int64)
        image = np.asarray(image, np.float32)

        # save_label(label,city_palette)
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        return image.copy(),label.copy().astype(np.int32),name


if __name__ == '__main__':
    dataset = cityscapesDataSet(r"./ADVNET/data/Cityscapes",
                                './src/dataset/cityscapes_list/val.txt',
                                crop_size=(1024, 512))
    dataset = ds.GeneratorDataset(dataset, column_names=['image', 'label', 'name'], shuffle=False)
    dataset = dataset.batch(batch_size=2)

    dataset_iterator = dataset.create_dict_iterator()
    for i ,data in enumerate(dataset_iterator):
        # data = dataset.__getitem__(i)
        # print(data)
        img = data['image']
        name = data['name']
        print(img.shape)
