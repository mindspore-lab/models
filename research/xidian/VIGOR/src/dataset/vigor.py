# Copyright 2024 Xidian University
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
from PIL import Image
import numpy as np
import os
import random

import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindspore.dataset import transforms
from mindspore.dataset.vision import Normalize
from mindspore.dataset.vision import Resize
from mindspore.dataset.vision import ToTensor


def sat_transform():
    return transforms.Compose([
        Resize(size=(640, 640)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225],
                  is_hwc=False),
    ])

def grd_transform():
    return transforms.Compose([
        Resize(size=(320, 640)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225],
                  is_hwc=False),
    ])

def get_dataloader(root='/path/data/VIGOR/', 
                   batch_size=32, num_workers=4, same_area=False):
    train_set = VIGOR(mode='train', root=root, same_area=same_area)
    val_sat_set = VIGOR(mode='test_reference', root=root, same_area=same_area)
    val_grd_set = VIGOR(mode='test_query', root=root, same_area=same_area)
    len_train, len_val_sat, len_val_grd = len(train_set), len(val_sat_set), len(val_grd_set)
    
    train_loader = GeneratorDataset(train_set, column_names=['sat', 'sat_semi', 'grd',
                                                             'delta', 'ratio'],
                                    shuffle=True, num_parallel_workers=num_workers)
    train_loader = train_loader.map(sat_transform(), 'sat')
    train_loader = train_loader.map(sat_transform(), 'sat_semi')
    train_loader = train_loader.map(grd_transform(), 'grd')
    train_loader = train_loader.batch(batch_size)
    
    val_sat_loader = GeneratorDataset(val_sat_set, column_names=['sat', 'idx'], 
                                      shuffle=False, num_parallel_workers=num_workers)
    val_sat_loader = val_sat_loader.map(sat_transform(), 'sat')
    val_sat_loader = val_sat_loader.batch(batch_size)
    
    val_grd_loader = GeneratorDataset(val_grd_set, column_names=['grd', 'idx', 'label'], 
                                      shuffle=False, num_parallel_workers=num_workers)
    val_grd_loader = val_grd_loader.map(grd_transform(), 'grd')
    val_grd_loader = val_grd_loader.batch(batch_size)

    return train_loader, val_sat_loader, val_grd_loader, len_train, len_val_sat, len_val_grd


# Same loader from VIGOR, modified for pytorch
class VIGOR():
    def __init__(self, mode = '', 
                 root = '/path/data/VIGOR/', 
                 same_area=False, print_bool=False, polar = '', args=None):
        super(VIGOR, self).__init__()

        self.args = args
        self.root = root
        self.polar = polar

        self.mode = mode
        self.sat_size = [320, 320]
        self.sat_size_default = [320, 320]
        self.grd_size = [320, 640]

        if print_bool:
            print(self.sat_size, self.grd_size)

        self.sat_ori_size = [640, 640]
        self.grd_ori_size = [320, 640]

        self.same_area = same_area
        label_root = 'splits'

        if same_area:
            self.train_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
            self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        else:
            self.train_city_list = ['NewYork', 'Seattle']
            self.test_city_list = ['SanFrancisco', 'Chicago']

        self.train_sat_list = []
        self.train_sat_index_dict = {}
        self.delta_unit = [0.0003280724526376747, 0.00043301140280175833]
        idx = 0
        # load sat list
        for city in self.train_city_list:
            train_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(train_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.train_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.train_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            if print_bool:
                print('InputData::__init__: load', train_sat_list_fname, idx)
        self.train_sat_list = np.array(self.train_sat_list)
        self.train_sat_data_size = len(self.train_sat_list)
        if print_bool:
            print('Train sat loaded, data size:{}'.format(self.train_sat_data_size))

        self.test_sat_list = []
        self.test_sat_index_dict = {}
        self.__cur_sat_id = 0  # for test
        idx = 0
        for city in self.test_city_list:
            test_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(test_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.test_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.test_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            if print_bool:
                print('InputData::__init__: load', test_sat_list_fname, idx)
        self.test_sat_list = np.array(self.test_sat_list)
        self.test_sat_data_size = len(self.test_sat_list)
        if print_bool:
            print('Test sat loaded, data size:{}'.format(self.test_sat_data_size))

        self.train_list = []
        self.train_label = []
        self.train_sat_cover_dict = {}
        self.train_delta = []
        idx = 0
        for city in self.train_city_list:
            # load train panorama list
            train_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_train.txt'
            if self.same_area else 'pano_label_balanced.txt')
            with open(train_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.train_sat_index_dict[data[i]])
                    label = np.array(label).astype(np.int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.train_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.train_label.append(label)
                    self.train_delta.append(delta)
                    if not label[0] in self.train_sat_cover_dict:
                        self.train_sat_cover_dict[label[0]] = [idx]
                    else:
                        self.train_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            if print_bool:
                print('InputData::__init__: load ', train_label_fname, idx)
        self.train_data_size = len(self.train_list)
        self.train_label = np.array(self.train_label)
        self.train_delta = np.array(self.train_delta)
        if print_bool:
            print('Train grd loaded, data_size: {}'.format(self.train_data_size))

        self.__cur_test_id = 0
        self.test_list = []
        self.test_label = []
        self.test_sat_cover_dict = {}
        self.test_delta = []
        idx = 0
        for city in self.test_city_list:
            # load test panorama list
            test_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_test.txt'
            if self.same_area else 'pano_label_balanced.txt')
            with open(test_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.test_sat_index_dict[data[i]])
                    label = np.array(label).astype(np.int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(np.float32)
                    self.test_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.test_label.append(label)
                    self.test_delta.append(delta)
                    if not label[0] in self.test_sat_cover_dict:
                        self.test_sat_cover_dict[label[0]] = [idx]
                    else:
                        self.test_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            if print_bool:
                print('InputData::__init__: load ', test_label_fname, idx)
        self.test_data_size = len(self.test_list)
        self.test_label = np.array(self.test_label)
        self.test_delta = np.array(self.test_delta)
        if print_bool:
            print('Test grd loaded, data size: {}'.format(self.test_data_size))

        self.train_sat_cover_list = list(self.train_sat_cover_dict.keys())

    def __getitem__(self, index):
        if 'train' in self.mode:
            idx = random.choice(self.train_sat_cover_dict[self.train_sat_cover_list[index%len(self.train_sat_cover_list)]])
            grd = Image.open(self.train_list[idx])
            delta = self.train_delta[idx, 0]
            
            sat = Image.open(self.train_sat_list[self.train_label[idx][0]]).convert('RGB')            
            
            # semi-positive images 
            randx = random.randrange(1, 4)
            sat_semi = Image.open(self.train_sat_list[self.train_label[idx][randx]]).convert('RGB')
            semi_delta = self.train_delta[idx, randx]
            
            ratio = self.distance_score(delta, semi_delta)
            
            return sat, sat_semi, grd, ms.tensor(delta, dtype=ms.float32), ms.tensor(ratio, dtype=ms.float32)
        elif 'test_reference' in self.mode:
            img_reference = Image.open(self.test_sat_list[index]).convert('RGB')
            return img_reference, ms.tensor(index, dtype=ms.int64)
        elif 'test_query' in self.mode:
            img_query = Image.open(self.test_list[index])
            return img_query, ms.tensor(index, dtype=ms.int64), ms.tensor(self.test_label[index][0], dtype=ms.int64)
        else:
            raise Exception('not implemented!!')

    def distance_score(self, delta_1, delta_2, mode='IOU', L=640.):
        if mode == 'distance':
            distance_1 = np.sqrt(delta_1[0] * delta_1[0] + delta_1[1] * delta_1[1])
            distance_2 = np.sqrt(delta_2[0] * delta_2[0] + delta_2[1] * delta_2[1])
            ratio = distance_1/distance_2
        elif mode == 'IOU':
            IOU_1 = 1. / (1. - (1 - np.abs(delta_1[0]) / L) * (1. - np.abs(delta_1[1]) / L) / 2.) - 1
            IOU_2 = 1. / (1. - (1 - np.abs(delta_2[0]) / L) * (1. - np.abs(delta_2[1]) / L) / 2.) - 1
            ratio = IOU_2/ IOU_1
        return ratio
    
    def __len__(self):
        if 'train' in self.mode:
            return len(self.train_sat_cover_list) * 2  # one aerial image has 2 positive queries
        elif 'test_reference' in self.mode:
            return len(self.test_sat_list)
        elif 'test_query' in self.mode:
            return len(self.test_list)
        else:
            print('not implemented!')
            raise Exception
