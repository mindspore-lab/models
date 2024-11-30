# Copyright 2021 Huawei Technologies Co., Ltd
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

"""Datasets"""
import os
import copy
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.vision import Border
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype
from model_utils.config import config


class MyDataset():
    """Self Defined dataset."""
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label
    
def load_train_lt(imb_ratio=100,batch_size=config.batch_size):
    # calculated the number of samples of each class  according to  imbalance ratio
    num_each_class = np.array(
        [int(np.floor(5000 * ((1 / imb_ratio) ** (1 / 9)) ** (i))) for i in range(10)])
    # load cifar10 dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data/cifar10_dataset_directory")
    dataset = ds.Cifar10Dataset(dataset_dir=data_path, shuffle=True, usage='train')
    images = []
    labels=[]
    num_each_class_cal = copy.deepcopy(num_each_class)
    for (image,label) in dataset:
        if num_each_class_cal[label.asnumpy()]>0:
            images.append(image)
            labels.append(label)
            num_each_class_cal[label.asnumpy()]-=1
    # build long-tail dataset
    data_lt = MyDataset(images,labels)
    dataset_lt = ds.GeneratorDataset(data_lt, column_names=["image", "label"],shuffle=True)
    random_crop_op = vision.RandomCrop(32, padding=4, padding_mode=Border.EDGE)
    RandomHorizontalFlip_op = vision.RandomHorizontalFlip()
    HWC2CHW_op = vision.HWC2CHW()
    Normalize_op = vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False)
    type_cast_op_data = transforms.TypeCast(mstype.float32)
    transform_data_list = [random_crop_op, HWC2CHW_op, type_cast_op_data, RandomHorizontalFlip_op, Normalize_op]
    type_cast_op = transforms.TypeCast(mstype.int32)
    dataset_lt = dataset_lt.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)
    dataset_lt = dataset_lt.map(operations=transform_data_list, input_columns="image", num_parallel_workers=1)
    dataset_lt = dataset_lt.batch(batch_size, drop_remainder=True)
    return num_each_class, dataset_lt


def load_test(batch_size=config.batch_size):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data/cifar10_dataset_directory")
    dataset = ds.Cifar10Dataset(dataset_dir=data_path, shuffle=False, usage='test')
    HWC2CHW_op = vision.HWC2CHW()
    Normalize_op = vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False)
    type_cast_op = transforms.TypeCast(mstype.int32)
    type_cast_op_data = transforms.TypeCast(mstype.float32)
    transform_data_list = [HWC2CHW_op,type_cast_op_data,Normalize_op]
    dataset = dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)
    dataset = dataset.map(operations=transform_data_list, input_columns="image", num_parallel_workers=1)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset



if __name__ == '__main__':
    # dataset = load_test(batch_size=64)
    num_cls, dataset_lt = load_train_lt(imb_ratio=10,batch_size=64)
    iterator = dataset_lt.create_dict_iterator()
    for data in iterator:
        image = data['image']
        label = data['label']
        print(image.shape)
        print(label.shape)
