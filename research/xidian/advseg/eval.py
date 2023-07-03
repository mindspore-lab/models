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

import mindspore
import mindspore.dataset as ds
import mindspore.ops as ops
import numpy as np
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size

from src.advnet.adaptsegnet import get_adaptsegnetCell
from src.dataset.cityscapes_dataset import cityscapesDataSet
from src.model_utils import config, get_device_id, evaluation


def main():
    """Create the model and start the evaluation process."""

    # args = get_arguments()
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    input_size = config.input_size_target
    output_size = config.output_size

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    if config.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
    elif config.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
    print('设备：', config.device_target)

    if config.is_distributed:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
        parallel_mode = mindspore.context.ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True,
                                          device_num=config.group_size)

    os.makedirs(config.save_result, exist_ok=True)

    # model = DeeplabMulti(num_classes=config.num_classes)
    model = get_adaptsegnetCell(config)

    print('model path:', config.restore_from)

    if config.restore_from:
        saved_state_dict = mindspore.load_checkpoint(config.restore_from)
        mindspore.load_param_into_net(model, saved_state_dict)
        print('success load model !')

    if  os.path.exists(os.path.join(config.data_dir_target, 'leftImg8bit_trainvaltest')):
        val_data_path = os.path.join(config.data_dir_target, 'leftImg8bit_trainvaltest')
    else:
        val_data_path = config.data_dir_target

    cityscapes_generator = cityscapesDataSet(val_data_path, os.path.join(config.data_list_target, 'val.txt'),
                                             crop_size=input_size, scale=False,
                                             mirror=False, mean=IMG_MEAN,
                                             set='val')
    cityscapes_dataset = ds.GeneratorDataset(cityscapes_generator, shuffle=False,
                                             column_names=['image', 'name'])
    cityscapes_dataset = cityscapes_dataset.batch(batch_size=1)
    target_iterator = cityscapes_dataset.create_dict_iterator()
    interp = ops.ResizeBilinear(size=output_size)
    evaluation(model=model.model_G,
               testloader=target_iterator,
               interp=interp,
               data_dir_target=config.data_dir_target,
               save_result=config.save_result,
               data_list_target=config.data_list_target, save=True,
               config=config)


if __name__ == '__main__':
    main()
