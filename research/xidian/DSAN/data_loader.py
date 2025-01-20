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
'''Dataloader'''
import os
from mindspore import dataset
from mindspore.dataset import transforms
from mindspore.dataset import vision
from mindspore import dtype as mstype


def load_training(root_path, dir, batch_size):
    data = dataset.ImageFolderDataset(dataset_dir=os.path.join(root_path, dir,'images'), shuffle=True, decode=True)
    transform_list = transforms.Compose(
        [vision.Resize([256, 256]),
         vision.RandomCrop(224),
         vision.RandomHorizontalFlip(),
         vision.ToTensor()])
    image_folder_dataset = data.map(operations=transform_list, input_columns="image")
    type_cast_op = transforms.TypeCast(mstype.int32)
    image_folder_dataset = image_folder_dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)
    image_folder_dataset = image_folder_dataset.batch(batch_size=batch_size,drop_remainder=True)
    return image_folder_dataset
    
def load_testing(root_path, dir, batch_size):
    data = dataset.ImageFolderDataset(dataset_dir=os.path.join(root_path, dir, 'images'), shuffle=True, decode=True)
    transform_list = transforms.Compose(
        [vision.Resize([224, 224]),
         vision.ToTensor()])
    image_folder_dataset = data.map(operations=transform_list, input_columns="image")
    type_cast_op = transforms.TypeCast(mstype.int32)
    image_folder_dataset = image_folder_dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)
    image_folder_dataset = image_folder_dataset.batch(batch_size=batch_size)
    return image_folder_dataset

def load_data(root_path, src, tar, batch_size):
    loader_src = load_training(root_path, src, batch_size)
    loader_tar = load_training(root_path, tar, batch_size)
    loader_tar_test = load_testing(
        root_path, tar, batch_size)
    return loader_src, loader_tar, loader_tar_test