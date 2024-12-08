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
"""
preprocess script
"""
import os
import mindspore.dataset as ds
from src.dataset.cityscapes_dataset import cityscapesDataSet
from src.model_utils import config
import numpy as np

def preprocess(result_path):
    input_size = config.input_size_target
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    if os.path.exists(os.path.join(config.data_dir_target, 'leftImg8bit_trainvaltest')):
        val_data_path = os.path.join(config.data_dir_target, 'leftImg8bit_trainvaltest')
    else:
        val_data_path = config.data_dir_target
    cityscapes_generator = cityscapesDataSet(val_data_path, os.path.join(config.data_list_target, 'val.txt'),
                                             crop_size=input_size, scale=False,
                                             mirror=False, mean=IMG_MEAN,
                                             set='val')
    cityscapes_dataset = ds.GeneratorDataset(cityscapes_generator, shuffle=False,
                                             column_names=['image', 'size','name'])
    cityscapes_dataset = cityscapes_dataset.batch(batch_size=1)



    img_path = os.path.join(result_path, "img_data")
    os.makedirs(img_path,exist_ok=True)
    for idx, data in enumerate(cityscapes_dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["image"]
        name = data['name'][0]
        file_name = 'advnet_1_'+name.split('/')[-1].replace('.png','.bin')
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)


    # label_path = os.path.join(result_path, "label")
    # os.makedirs(label_path)
    #
    # for idx, data in enumerate(cityscapes_dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
    #     img_data = data["image"]
    #     img_label = data["label"]
    #
    #     file_name = "google_cifar10_" + str(config.batch_size) + "_" + str(idx) + ".bin"
    #     img_file_path = os.path.join(img_path, file_name)
    #     img_data.tofile(img_file_path)
    #
    #     label_file_path = os.path.join(label_path, file_name)
    #     img_label.tofile(label_file_path)

    print("=" * 20, "export bin files finished", "=" * 20)

if __name__ == '__main__':
    preprocess(config.output_path)
