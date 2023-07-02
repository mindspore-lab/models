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
"""Adversarial adaptation to train target encoder."""
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score
from mindspore import ops
import mindspore as ms
import mindspore.nn as nn
from models.lenet import LeNetClassifier, LeNetEncoder
from utils import get_data_loader
from model_utils.config import config


def eval_tgt(encoder, classifier, data_loader):
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


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################
    loss_tgt_encoder = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    loss_critic = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    optimizer_tgt_encoder = nn.Adam(tgt_encoder.trainable_params(),
                                    learning_rate=config.c_learning_rate,
                                    beta1=config.beta1,
                                    beta2=config.beta2)
    optimizer_critic = nn.Adam(critic.trainable_params(),
                               learning_rate=config.d_learning_rate,
                               beta1=config.beta1,
                               beta2=config.beta2)

    class WithLossCellG(nn.Cell):
        def __init__(self, net_d, net_g, loss_fn):
            super(WithLossCellG, self).__init__(auto_prefix=True)
            self.net_d = net_d
            self.net_g = net_g
            self.loss_fn = loss_fn

        def construct(self, tgt_images):
            tge_feat = self.net_g(tgt_images)
            out = self.net_d(tge_feat)
            label_tgt = ms.numpy.ones(tge_feat.shape[0]).astype(np.int32)
            loss = self.loss_fn(out, label_tgt)
            return loss

    class WithLossCellD(nn.Cell):
        def __init__(self, net_d, net_g, src_encoder, loss_fn):
            super(WithLossCellD, self).__init__(auto_prefix=True)
            self.net_d = net_d
            self.net_g = net_g
            self.loss_fn = loss_fn
            self.src_encoder = src_encoder

        def construct(self, src_images, tgt_images):
            src_feat = self.src_encoder(src_images)
            tgt_feat = self.net_g(tgt_images)
            tgt_feat = ops.stop_gradient(tgt_feat)
            op = ops.Concat()
            feat_concat = op((src_feat, tgt_feat))
            pred_concat = critic(feat_concat)
            label_src = ms.numpy.ones(src_feat.shape[0]).astype(np.int32)
            label_tgt = ms.numpy.zeros(tgt_feat.shape[0]).astype(np.int32)
            label_concat = op((label_src, label_tgt))
            loss_critic_output = self.loss_fn(pred_concat, label_concat)
            return loss_critic_output

    class DCGAN(nn.Cell):
        def __init__(self, my_train_one_step_cell_for_d, my_train_one_step_cell_for_g):
            super(DCGAN, self).__init__(auto_prefix=True)
            self.my_train_one_step_cell_for_d = my_train_one_step_cell_for_d
            self.my_train_one_step_cell_for_g = my_train_one_step_cell_for_g

        def construct(self, real_data, latent_code):
            output_d = self.my_train_one_step_cell_for_d(real_data, latent_code).view(-1)
            net_d_loss = output_d.mean()
            output_g = self.my_train_one_step_cell_for_g(latent_code).view(-1)
            net_g_loss = output_g.mean()
            return net_d_loss, net_g_loss

    net_d_with_criterion = WithLossCellD(critic, tgt_encoder, src_encoder, loss_critic)
    net_g_with_criterion = WithLossCellG(critic, tgt_encoder, loss_tgt_encoder)

    my_train_one_step_cell_for_d = nn.TrainOneStepCell(net_d_with_criterion, optimizer_critic)
    my_train_one_step_cell_for_g = nn.TrainOneStepCell(net_g_with_criterion, optimizer_tgt_encoder)
    dcgan = DCGAN(my_train_one_step_cell_for_d, my_train_one_step_cell_for_g)

    dcgan.set_train()
    tgt_data_loader_eval = get_data_loader(config.tgt_dataset, train=False)
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_classifier_dict = ms.load_checkpoint(
        os.path.join(current_dir, config.model_root, config.src_classifier_checkpoint))
    src_classifier = LeNetClassifier()
    ms.load_param_into_net(src_classifier, src_classifier_dict)
    step = 0
    best_accuracy = 0
    ####################
    # 2. train network #
    ####################
    time_tgt_train_onestep_avg_list = []
    for epoch in range(config.num_epochs):
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        time_tgt_train_oneepoch_begin = time.perf_counter()
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            net_d_loss, net_g_loss = dcgan(images_src, images_tgt)

        time_tgt_train_oneepoch_end = time.perf_counter()
        time_tgt_train_oneepoch = (time_tgt_train_oneepoch_end - time_tgt_train_oneepoch_begin) * 1000
        time_tgt_train_onestep_avg = time_tgt_train_oneepoch / (step + 1)
        print("Epoch [{}/{}] Step [{}] d_loss={:.5f} g_loss={:.5f} \n"
              "time_tgt_train_oneepoch:{:.3f}ms time_tgt_train_onestep_avg:{:.3f}ms"
              .format(epoch + 1,
                      config.num_epochs,
                      step + 1,
                      ms.numpy.average(net_d_loss).asnumpy(),
                      ms.numpy.average(net_g_loss).asnumpy(),
                      time_tgt_train_oneepoch,
                      time_tgt_train_onestep_avg
                      ))
        accuracy = eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
        # save best model parameters
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            ms.save_checkpoint(tgt_encoder, os.path.join(current_dir, config.model_root, config.tgt_encoder_checkpoint))
        if epoch > 6:
            time_tgt_train_onestep_avg_list.append(time_tgt_train_onestep_avg)
        if epoch + 1 % config.save_step == 0:
            ms.save_checkpoint(critic,
                               os.path.join(current_dir, config.model_root, "ADDA-critic-{}.ckpt".format(epoch + 1)))
            ms.save_checkpoint(tgt_encoder, os.path.join(current_dir, config.model_root,
                                                         "ADDA-target-encoder-{}.ckpt".format(epoch + 1)))

    tgt_encoder_dict = ms.load_checkpoint(os.path.join(current_dir, config.model_root, config.tgt_encoder_checkpoint))
    tgt_encoder = LeNetEncoder()
    ms.load_param_into_net(tgt_encoder, tgt_encoder_dict)
    print("=== Train on source data is finished ===")
    print("time of onestep train on target data:{:.3f}ms".format(np.mean(time_tgt_train_onestep_avg_list)))
    print("--------------best_accuracy-------------")
    print(best_accuracy)
    return tgt_encoder
