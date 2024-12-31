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
    """Create the model and start the training."""
    """Part 1: Environment Preparation"""
    # 以下三个顺序不可乱，如果顺序混乱，可能会导致一些问题
    set_debug(config)
    set_environment(config)
    platform_preprocess(config)
    mindspore.context.set_context(enable_graph_kernel=False)
    print("Please check the above information for the configurations", flush=True)
    print(config)

    config.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    """Part 2: dataset Preparation"""
    gta_dataset = GTA5DataSet(config.data_dir, config.data_list,
                              max_iters=config.num_steps,
                              crop_size=config.input_size, mean=config.IMG_MEAN)
    cityscapes_dataset = cityscapesDataSet(config.data_dir_target, os.path.join(config.data_list_target, 'train.txt'),
                                           max_iters=config.num_steps,
                                           crop_size=config.input_size_target,
                                           mean=config.IMG_MEAN,
                                           set='train')
    if config.device_num > 1:
        sampler = ds.DistributedSampler(shuffle=True, shard_id=config.device_id, num_shards=config.device_num, num_samples=config.num_steps)
        gta_dataset = ds.GeneratorDataset(gta_dataset, ["s_image", "s_label"], sampler=sampler)
        sampler = ds.DistributedSampler(shuffle=True, shard_id=config.device_id, num_shards=config.device_num, num_samples=config.num_steps)
        cityscapes_dataset = ds.GeneratorDataset(cityscapes_dataset, ["t_image", "None","name"], sampler=sampler)
    else:
        sampler = ds.RandomSampler(num_samples=config.num_steps)
        gta_dataset = ds.GeneratorDataset(gta_dataset, ["s_image", "s_label"], sampler=sampler)
        sampler = ds.RandomSampler(num_samples=config.num_steps)
        cityscapes_dataset = ds.GeneratorDataset(cityscapes_dataset, ["t_image", "None","name"], sampler=sampler)

    train_dataset = ds.zip((gta_dataset, cityscapes_dataset))
    train_dataset = train_dataset.project(['s_image', 't_image', 's_label'])
    train_dataset = train_dataset.batch(batch_size=config.batch_size)

    print('GTA5 Train Data Path:\t', config.data_dir)
    print("Cityscapes Train Data path:\t", config.data_dir_target)
    print('the length of dataset is {}'.format(train_dataset.get_dataset_size()))
    print('the batch size is {}'.format(train_dataset.batch_size))

    val_dataset = cityscapesDataSet(config.data_dir_target, os.path.join(config.data_list_target, 'val.txt'),
                                crop_size=config.input_size_target, mean=config.IMG_MEAN,set='val')
    val_dataset = ds.GeneratorDataset(val_dataset, shuffle=False, column_names=['image', "label",'name'],)
    val_dataset = val_dataset.batch(batch_size=1)

    # [Part 2: net]
    net = get_adaptsegnetCell(config)

    print('Parameter Number:')
    print('Generator       :\t{}'.format(len(net.net_G.trainable_params())))
    print('Discriminator_1 :\t{}'.format(len(net.net_D1.trainable_params())))
    print('Discriminator_2 :\t{}'.format(len(net.net_D2.trainable_params())))
    print('All:            :\t{}'.format(len(net.trainable_params())))
    # [Part 3: Optimizer and Loss function]

    calc_loss = SoftmaxCrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    #  [Whether continue train]

    steps_per_epoch = 10
    step_cb = StepMonitor(per_print_times=steps_per_epoch)
    ckpt_cb = CheckpointMonitor(config, net, val_dataset)
    cb = [step_cb, ckpt_cb]
    train_net = get_TrainOneStepCell(config, net, calc_loss, bce_loss)

    model = Model(train_net)

    model.train(1, train_dataset, callbacks=cb, dataset_sink_mode=False)

    adv_target1 = [0.0002, 0.0001, 0.001, 0.0005, 0.0003]
    adv_target2 = [0.001]

    for adv_1 in adv_target1:
        for adv_2 in adv_target2:
            config.lambda_[1]=adv_1
            config.lambda_[2]=adv_2
            model.train(1, train_dataset, callbacks=cb, dataset_sink_mode=False)
    print('Train Over ! Save Over model ')


if __name__ == '__main__':
    main()
