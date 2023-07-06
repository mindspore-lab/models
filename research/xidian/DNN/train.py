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
import ast
import argparse
import mindspore
from mindspore import context, Model, Tensor
import os
from src.model.nn import *
from src.dataset import create_loaders
import mindspore.nn as nn
import numpy as np
from model_utils.config import config


def get_lr(base_lr, total_epochs, steps_per_epoch, global_epoch=0):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    global_steps = steps_per_epoch * global_epoch
    for i in range(total_steps):
        lr = base_lr * (1.0 - float(i) * float(config.batchsize) / (config.num_train * float(config.NN_numepochs)))
        lr_each_step.append(lr)
    lr_each_step = lr_each_step[global_steps:]
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


if __name__ == '__main__':
    context.set_context(mode=1, device_target=config.device_target)  #GRAPH_MODE(0) or PYNATIVE_MODE(1)
    if config.modelArts_mode:
        import moxing as mox
        local_data_url = '/cache/data'
        local_train_url = '/cache/ckpt'
        # download dataset from obs to cache
        mox.file.copy_parallel(src_url=config.data_url, dst_url=local_data_url,device_id=config.device_id)
        config.dataroot = local_data_url
        LOG_DIR = local_train_url
        ckpt_save_dir = local_train_url

    train_data=create_loaders(config)
    step_size = train_data.get_dataset_size()
    rbm=DBN_pretrain(config,train_data)
    net=Model_NN(in_dim=1568,init_weight=rbm)
    lr = Tensor(get_lr(base_lr=config.lr, total_epochs=config.NN_numepochs, steps_per_epoch=step_size))
    net_opt = nn.SGD(net.trainable_params(), learning_rate=lr, momentum=0.9, dampening=0.9, weight_decay=config.wd)
    loss = nn.MSELoss()
    train_net=ModelWithLossCell(net,loss)
    model = Model(network=train_net, optimizer=net_opt)
    print('start nn train')
    train_loop(net, train_data, loss, net_opt,config)
    mindspore.save_checkpoint(net, "nn.ckpt")




