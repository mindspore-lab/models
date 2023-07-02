# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Pre-train encoder and classifier for source dataset."""
import os
import time
from sklearn.metrics import accuracy_score
import mindspore.nn as nn
import mindspore as ms
import numpy as np
from utils import get_data_loader
from model_utils.config import config


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    acc_list = []
    for data in data_loader.create_dict_iterator():
        images, labels = data['image'], data['label']
        preds = classifier(encoder(images))
        labels_cls = labels.argmax(1)
        pred_cls = preds.argmax(1)
        acc = accuracy_score(pred_cls.asnumpy(), labels_cls.asnumpy())
        acc_list.append(acc)

    acc_avg = np.mean(acc_list)
    print(" Avg Accuracy = {:2%}".format(acc_avg))
    return acc_avg


def train_src(encoder, classifier, data_loader):
    """Train classifier for source domain."""

    ####################
    # 1. setup network #
    ####################
    class MyWithLossCell(nn.Cell):

        def __init__(self, encoder, classifier, loss_fn):
            super(MyWithLossCell, self).__init__(auto_prefix=True)
            self.encoder = encoder
            self.classifier = classifier
            self.loss_fn = loss_fn

        def construct(self, image, label):
            output = self.classifier(self.encoder(image))
            loss = self.loss_fn(output, label)
            return loss

    class NET(nn.Cell):
        def __init__(self, my_train_one_step_cell):
            super(NET, self).__init__(auto_prefix=True)
            self.my_train_one_step_cell = my_train_one_step_cell

        def construct(self, image, label):
            output = self.my_train_one_step_cell(image, label).view(-1)
            loss = output.mean()
            return loss

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = nn.Adam(
        list(encoder.trainable_params()) + list(classifier.trainable_params()),
        learning_rate=config.c_learning_rate,
        beta1=config.beta1,
        beta2=config.beta2)
    src_data_loader_eval = get_data_loader(config.src_dataset, train=False)  # 加载测试数据
    loss_net = MyWithLossCell(encoder, classifier, loss)
    my_train_one_step_cell = nn.TrainOneStepCell(loss_net, optimizer)
    net = NET(my_train_one_step_cell)
    net.set_train()
    acc_best = 0
    time_src_train_onestep_avg_list = []
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ####################
    # 2. train network #
    ####################
    for epoch in range(config.num_epochs_pre):
        time_src_train_oneepoch_begin = time.perf_counter()
        step = 0
        for data in data_loader.create_dict_iterator():
            image, label = data['image'], data['label']
            loss_output = net(image, label)
            if (step + 1) % config.log_step_pre == 0:
                print("Epoch [{}/{}] Step [{}]: loss={}"
                      .format(epoch + 1,
                              config.num_epochs_pre,
                              step + 1,
                              ms.numpy.average(loss_output)))
            step += 1
        time_src_train_oneepoch_end = time.perf_counter()
        time_src_train_oneepoch = (time_src_train_oneepoch_end - time_src_train_oneepoch_begin) * 1000
        time_src_train_onestep_avg = time_src_train_oneepoch / (step + 1)
        if epoch + 1 > 1:
            time_src_train_onestep_avg_list.append(time_src_train_onestep_avg)
        print("Epoch{}/{} time_src_train_oneepoch:{:.3f}ms time_src_train_onestep_avg:{:.3f}ms"
              .format(epoch + 1,
                      config.num_epochs_pre,
                      time_src_train_oneepoch,
                      time_src_train_onestep_avg))
        if (epoch + 1) % config.eval_step_pre == 0:
            acc = eval_src(encoder, classifier, src_data_loader_eval)
            # save best model parameters
            if acc >= acc_best:
                acc_best = acc
                ms.save_checkpoint(encoder, os.path.join(current_dir, config.model_root, config.src_encoder_checkpoint))
                ms.save_checkpoint(classifier,
                                   os.path.join(current_dir, config.model_root, config.src_classifier_checkpoint))
        if (epoch + 1) % config.save_step_pre == 0:
            ms.save_checkpoint(encoder, os.path.join(current_dir, config.model_root,
                                                     "ADDA-source-encoder-{}.ckpt".format(epoch + 1)))
            ms.save_checkpoint(classifier, os.path.join(current_dir, config.model_root,
                                                        "ADDA-source-classifier-{}.ckpt".format(epoch + 1)))
    print("=== Train on source data is finished ===")
    print("time of onestep train on source data:{:.3f}ms".format(np.mean(time_src_train_onestep_avg_list)))
    return encoder, classifier


if __name__ == '__main__':
    print(os.getcwd())
