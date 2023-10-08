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
import scipy.io as scio
import mindspore.dataset as ds
import numpy as np
from model_utils.config import config
def normalization(data):
    M_m = np.max(data) - np.min(data)
    return (data - np.min(data)) / M_m
class dataset():
    def __init__(self,):
        super(dataset, self).__init__()
        print('loading data---')
        self.mat_path = config.dataroot+'/huanghepatch14.mat'
        self.mat_label_path = config.dataroot+'/huanghepatchlabel14.mat'
        self.load_mat = scio.loadmat(self.mat_path)
        self.load_label_mat = scio.loadmat(self.mat_label_path)
        self.data=self.load_mat['patch']
        self.data=(normalization(self.data)*253+1)/255
        self.label_data=self.load_label_mat['patchlabel']
        print('loading data done!!!')
    def __getitem__(self, index):
        img = self.data[index].astype('float32')
        label = self.label_data[index]
        return img ,label
    def __len__(self):
        return len(self.label_data)
def create_loaders(args = None):
    train_dataset=ds.GeneratorDataset(dataset(),
                                      column_names=["data", "label"],
                                      )
    train_dataset = train_dataset.batch(args.batchsize)
    return train_dataset
