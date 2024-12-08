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
import numpy as np
from mindspore import context, load_checkpoint, load_param_into_net, export
import mindspore.dataset as ds

from src.advnet import get_deeplab_v2
from src.model_utils import split_checkpoint
from src.model_utils import config
from src.model_utils import get_device_id
from src.dataset.cityscapes_dataset import cityscapesDataSet




def run_export():
    input_size = config.input_size_target
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=get_device_id())

    net_G = get_deeplab_v2(config.num_classes)
    h, w = config.input_size_target

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

    checkpoint = load_checkpoint(config.restore_from)

    param_G = split_checkpoint(checkpoint, ['net_G', 'net_D1', 'net_D2'])
    load_param_into_net(net_G, param_G['net_G'])
    config.file_name = config.file_name + '/'+os.path.split(config.restore_from)[1].replace('.ckpt', '')
    print("The export path:\t{}".format(config.file_name))

    export(net_G, cityscapes_dataset, file_name=config.file_name, file_format=config.file_format)
    print('Export Done !')

if __name__ == '__main__':
    config.restore_from = ""
    run_export()
