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
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops as ops

from model import dataloader
from model_utils.config import config
from model.utils import eval_generator

class WithLossCellG(nn.Cell):
    def __init__(self, net_e, net_c, loss_fn):
        super(WithLossCellG, self).__init__(auto_prefix=True)
        self.net_e = net_e
        self.net_c = net_c
        self.loss_fn = loss_fn

    def construct(self, data, label):
        encoder_out = self.net_e(data)
        class_out = self.net_c(encoder_out)
        loss = self.loss_fn(class_out, label)
        return loss


class Generator(nn.Cell):
    def __init__(self, train_for_g):
        super(Generator, self).__init__(auto_prefix=True)
        self.train_for_g = train_for_g

    def construct(self, data, label):
        out = self.train_for_g(data, label)
        net_g_loss = out.mean()
        return net_g_loss


def train_step1(encoder, classifier, train_dataloader, loss_fn, optimizer_ce):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dataloader = dataloader.mnist_dataloader(
        batch_size=config.batch_size, train=False)

    net_g_with_criterion = WithLossCellG(encoder, classifier, loss_fn)
    train_for_generator = nn.TrainOneStepCell(
        net_g_with_criterion, optimizer_ce)
    generator = Generator(train_for_generator)
    generator.set_train()

    pre_acc = 0
    
    
    
    for epoch in range(config.n_epoch_1):
        for batch, (data, label) in enumerate(train_dataloader):
            loss = generator(data, label)
            if batch % 100 == 0:
                print("Epoch-{}  iter-{}  Loss:{:.6f}".format(epoch,
                      batch, loss.asnumpy()))

        # test
        accuracy = eval_generator(encoder, classifier, test_dataloader)
        print("step1----Epoch %d/%d  accuracy: %.3f " %
              (epoch+1, config.n_epoch_1, accuracy))
        if accuracy > pre_acc:
            ms.save_checkpoint(encoder, os.path.join(
                current_dir, config.model_root, config.src_encoder_checkpoint))
            ms.save_checkpoint(classifier, os.path.join(
                current_dir, config.model_root, config.src_classifier_checkpoint))
    return encoder, classifier
