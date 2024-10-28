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
import random

import mindspore
import mindspore.dataset as ds
from mindspore import nn, Model
from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import LossMonitor, TimeMonitor

from src.model_utils import config, split_checkpoint

from src.dataset import get_dataset
from src.dataset.gta5_dataset import GTA5DataSet
from src.dataset.cityscapes_dataset import cityscapesDataSet

from src.advnet.adaptsegnet import get_adaptsegnetCell, get_TrainOneStepCell

from src.utils.loss import get_loss, SoftmaxCrossEntropyLoss
from src.utils.optimizer import get_optimizer
from src.utils.callbacks import StepMonitor, CheckpointMonitor
from src.utils.platform_process import platform_preprocess, platform_postprocess
from src.utils.set_environment import set_environment, cast_amp
from src.utils.set_debug import set_debug
def main():
    """Create the model and start the evaluation process."""
    """Part 1: Environment Preparation"""
    # 以下三个顺序不可乱，如果顺序混乱，可能会导致一些问题
    config.device_id=7
    set_debug(config)
    set_environment(config)
    platform_preprocess(config)
    mindspore.context.set_context(enable_graph_kernel=False)
    print("Please check the above information for the configurations", flush=True)
    print(config)

    config.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    os.makedirs(config.save_result, exist_ok=True)

    # model = DeeplabMulti(num_classes=config.num_classes)
    net = get_adaptsegnetCell(config)

    print('model path:', config.restore_from)

    # if config.restore_from:
    #     saved_state_dict = mindspore.load_checkpoint(config.restore_from)
    #     mindspore.load_param_into_net(net, saved_state_dict)
    #     print('success load model !')


    val_dataset = cityscapesDataSet(config.data_dir_target, os.path.join(config.data_list_target, 'val.txt'),
                                    crop_size=config.input_size_target, mean=config.IMG_MEAN, set='val')
    val_dataset = ds.GeneratorDataset(val_dataset, shuffle=False, column_names=['image', "label", 'name'], )
    val_dataset = val_dataset.batch(batch_size=1)

    ckpt_cb = CheckpointMonitor(config, net, val_dataset)
    miou = ckpt_cb.evaluation(save_mask=True)
    print("The iou is {}".format(miou))




if __name__ == '__main__':
    main()
