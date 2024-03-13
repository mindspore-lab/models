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
import os
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops as ops
from mindspore import Tensor as Tensor

from model import dataloader
from model_utils.config import config


class WithLossCellD(nn.Cell):
    def __init__(self, discriminator, loss_fn):
        super(WithLossCellD, self).__init__(auto_prefix=True)
        self.discriminator = discriminator
        self.loss_fn = loss_fn

    def construct(self, data, label):
        out = self.discriminator(data)
        loss = self.loss_fn(out, label)
        return loss


class Discriminator(nn.Cell):
    def __init__(self, train_for_dcd):
        super(Discriminator, self).__init__(auto_prefix=True)
        self.train_for_dcd = train_for_dcd

    def construct(self, data, label):
        out = self.train_for_dcd(data, label)
        net_d_loss = out.mean()
        return net_d_loss


def train_step2(encoder, discriminator, loss_fn, optimizer_dcd):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    net_d_with_criterion = WithLossCellD(discriminator, loss_fn)
    train_for_dcd = nn.TrainOneStepCell(net_d_with_criterion, optimizer_dcd)
    dcd = Discriminator(train_for_dcd)
    dcd.set_train()

    X_s, Y_s = dataloader.sample_data()
    X_t, Y_t = dataloader.create_target_samples(config.n_target_samples)
    min_loss = 1e9
    for epoch in range(config.n_epoch_2):
        groups, aa = dataloader.sample_groups(X_s, Y_s, X_t, Y_t, seed=epoch)
        n_iters = 4 * len(groups[1])
        index_list = np.random.permutation(n_iters)
        mini_batch_size = 40  # use mini_batch train can be more stable
        loss_mean = []
        X1, X2, ground_truths = [], [], []
        for index in range(n_iters):
            ground_truth = index_list[index]//len(groups[1])
            x1, x2 = groups[ground_truth][index_list[index] -
                                          len(groups[1])*ground_truth]
            X1.append(x1)
            X2.append(x2)

            ground_truths.append(ground_truth)
            # select data for a mini-batch to train
            if (index+1) % mini_batch_size == 0:
                X1, X2 = ops.stack(X1), ops.stack(X2)
                ground_truths = Tensor(ground_truths, dtype=ms.int32)
                X_cat = ops.Concat(axis=1)([encoder(X1), encoder(X2)]).copy()
                loss_dcd = dcd(X_cat, ground_truths)
                loss_mean.append(loss_dcd.asnumpy().item())
                X1, X2, ground_truths = [], [], []
        print("step2----Epoch %d/%d loss:%.3f" %
              (epoch+1, config.n_epoch_2, np.mean(loss_mean)))

        if np.mean(loss_mean) < min_loss: # save min loss checkpoints
            min_loss = np.mean(loss_mean)
            ms.save_checkpoint(discriminator, os.path.join(
                current_dir, config.model_root, config.tgt_discriminator_checkpoint))
