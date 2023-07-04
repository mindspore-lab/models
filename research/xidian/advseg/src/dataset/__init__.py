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
import mindspore.dataset as ds
import numpy as np
import os
from src.dataset.gta5_dataset import GTA5DataSet
from src.dataset.cityscapes_dataset import cityscapesDataSet


class PairDataset():
    def __init__(self, dataset_gta: GTA5DataSet, dataet_cityscapes: cityscapesDataSet):
        self.dataset_gta = dataset_gta
        self.dataset_cityscapes = dataet_cityscapes
        self.__len = min(dataset_gta.__len__(), dataet_cityscapes.__len__())

    def __len__(self):
        return self.__len

    def __getitem__(self, item):
        s_image, s_label = self.dataset_gta.__getitem__(item)
        t_image, _ = self.dataset_cityscapes.__getitem__(item)
        # image = np.concatenate([s_image, t_image], axis=0)
        # return image, s_label
        return s_image, t_image, s_label


def get_dataset(config, mode='train') -> ds.GeneratorDataset:
    if os.path.exists(os.path.join(config.data_dir_target, 'leftImg8bit_trainvaltest')):
        train_path = os.path.join(config.data_dir_target, 'leftImg8bit_trainvaltest')
    else:
        train_path = config.data_dir_target

    if mode == 'train':
        gta_dataset = GTA5DataSet(config.z, config.data_list,
                                  max_iters=config.num_steps,
                                  crop_size=config.input_size, scale=False,
                                  mirror=False, mean=config.IMG_MEAN)
        config.num_steps = gta_dataset.__len__()

        cityscapes_dataset = cityscapesDataSet(train_path, os.path.join(config.data_list_target, f'{config.set}.txt'),
                                               max_iters=config.num_steps,
                                               crop_size=config.input_size_target, scale=False,
                                               mirror=False, mean=config.IMG_MEAN,
                                               set=config.set)
        print('GTA5 Train Data Path:\t', config.data_dir)
        print("Cityscapes Train Data path:\t", train_path)
        dataset = PairDataset(gta_dataset, cityscapes_dataset)
        dataset = ds.GeneratorDataset(dataset, shuffle=True, column_names=['s_iamge', 't_image', 's_labels'])
        dataset = dataset.batch(batch_size=config.batch_size)

    else:
        dataset = cityscapesDataSet(train_path, os.path.join(config.data_list_target, mode + '.txt'),
                                    crop_size=config.input_size_target, scale=False,
                                    mirror=False, mean=config.IMG_MEAN,
                                    set=mode)
        dataset = ds.GeneratorDataset(dataset, shuffle=False, column_names=['image', 'name'],
                                      # num_shards=config.group_size,
                                      # shard_id=config.rank
                                      )
        dataset = dataset.batch(batch_size=1)
    return dataset


if __name__ == '__main__':
    from src.model_utils import config

    dataset = get_dataset(config)
