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

import numpy as np
import random

import mindspore
from mindspore import nn, Model
from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import LossMonitor, TimeMonitor

from src.model_utils import config, split_checkpoint

# from src.advnet import get_adaptsegnet, get_TrainOneStepCell
from src.advnet.adaptsegnet import get_adaptsegnetCell, get_TrainOneStepCell
from src.utils.loss import get_loss, SoftmaxCrossEntropyLoss
from src.utils.optimizer import get_optimizer
from src.utils.callbacks import StepMonitor, CheckpointMonitor
from src.dataset import get_dataset

print(config)

mindspore.common.set_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)


def main():
    """Create the model and start the training."""
    # config.debug = True

    config.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    # if config.model_arts:
    #     import moxing as mox
    #     local_data_url = '/cache/data'
    #     local_train_url = '/cache/ckpt'
    #     mox.file.copy_parallel(src_url=config.data_dir, dst_url=os.path.join(local_data_url, 'GTA5'))
    #     if os.path.exists(os.path.join(config.data_dir_target,'leftImg8bit_trainvaltest')) and \
    #        os.path.exists(os.path.join(config.data_dir_target,'gtFine_trainvaltest')):
    #         mox.file.copy_parallel(src_url=os.path.join(config.data_dir_target,'leftImg8bit_trainvaltest','leftImg8bit'), dst_url=os.path.join(local_data_url, 'Cityscapes/leftImg8bit'))
    #         mox.file.copy_parallel(src_url=(os.path.join(config.data_dir_target,'gtFine_trainvaltest','gtFine')), dst_url=os.path.join(local_data_url, 'Cityscapes/gtFine'))
    #     else:
    #         mox.file.copy_parallel(src_url=config.data_dir_target, dst_url=os.path.join(local_data_url, 'Cityscapes'))
    #     mox.file.copy_parallel(src_url=config.restore_from, dst_url=os.path.join(local_data_url, 'Pretrain_DeeplabMulti.ckpt'))
    #     # download dataset from obs to cache
    #     # if "obs://" in config.checkpoint_path:
    #     #     local_checkpoint_url = "/cache/" + config.checkpoint_path.split("/")[-1]
    #     #     mox.file.copy_parallel(config.checkpoint_path, local_checkpoint_url)
    #     #     config.checkpoint_path = local_checkpoint_url
    #     config.data_dir = os.path.join(local_data_url, 'GTA5')
    #     config.data_dir_target = os.path.join(local_data_url, 'Cityscapes')
    #     config.restore_from = os.path.join(local_data_url, 'Pretrain_DeeplabMulti.ckpt')
    #     # ckpt_save_dir = local_train_url + config.training_set

    # 环境配置

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    device_id = int(os.environ["DEVICE_ID"])
    if config.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
    elif config.device_target == "Ascend":
        context.set_context(device_id=device_id)
    print('设备：', config.device_target)

    if int(os.environ.get('DEVICE_NUM', 1)) > 1:
        config.is_distributed = True

    if config.is_distributed:
        init()
        config.rank = get_rank()
        config.group_size = int(os.environ.get('DEVICE_NUM', 8))
        parallel_mode = mindspore.context.ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=False,
                                          device_num=config.group_size,
                                          # parameter_broadcast=True
                                          )
        config.save_pred_every = config.save_pred_every // config.group_size

    if config.debug:
        config.batch_size = 2
        config.input_size = (128, 300)
        config.input_size_target = (128, 400)
        config.save_pred_every = 10
        config.num_steps_stop = 100
        config.not_val = False
        config.num_steps = 1000
        config.max_iters = 20000

    # [Part 1: dataset]
    dataset = get_dataset(config)
    print('the length of dataset is {}'.format(dataset.get_dataset_size()))
    # [Part 2: net]
    net = get_adaptsegnetCell(config)

    print('Parameter Number:')
    print('Generator       :\t{}'.format(len(net.model_G.trainable_params())))
    print('Discriminator_1 :\t{}'.format(len(net.model_D1.trainable_params())))
    print('Discriminator_2 :\t{}'.format(len(net.model_D2.trainable_params())))
    print('All:            :\t{}'.format(len(net.trainable_params())))
    # [Part 3: Optimizer and Loss function]

    # optimizer = get_optimizer(config, net)
    calc_loss = SoftmaxCrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    #  [Whether continue train]

    steps_per_epoch = 10
    step_cb = StepMonitor(per_print_times=steps_per_epoch)
    ckpt_cb = CheckpointMonitor(config, net, get_dataset(config, mode='val'))
    cb = [step_cb, ckpt_cb]
    # start train
    train_net = get_TrainOneStepCell(config, net, calc_loss, bce_loss)
    model = Model(train_net)
    # model.build(dataset, sink_size=config.save_pred_every)
    model.train(1, dataset, callbacks=cb, dataset_sink_mode=False)

    print('Train Over ! Save Over model ')


if __name__ == '__main__':
    main()
