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
from model.utils import eval_generator
from model_utils.config import config


def train_step3(encoder,classifier,discriminator,test_dataloader,loss_fn):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    X_s,Y_s=dataloader.sample_data()
    X_t,Y_t=dataloader.create_target_samples(config.n_target_samples)
  
    params_CE = list(classifier.trainable_params()) + list(encoder.trainable_params())
    optimizer_CE = ms.nn.Adam(params_CE,learning_rate=config.CE_lr_3)
    params_D = list(discriminator.trainable_params())
    optimizer_D = ms.nn.Adam(params_D,learning_rate=config.D_lr_3)

    net_CE_with_criterion = WithLossCellCE(classifier,encoder,discriminator,loss_fn)
    train_for_CE = nn.TrainOneStepCell(net_CE_with_criterion, optimizer_CE)
    ce = CE(train_for_CE)

    net_DD_with_criterion = WithLossCellDD(encoder,discriminator,loss_fn)
    train_for_DD = nn.TrainOneStepCell(net_DD_with_criterion, optimizer_D)
    dd = DD(train_for_DD)

    pre_acc = 0
    for epoch in range(config.n_epoch_3):
        ce.set_train()
        dd.set_train()
        groups, groups_y = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=config.n_epoch_3+epoch)
        G1, G2, G3, G4 = groups
        Y1, Y2, Y3, Y4 = groups_y
        groups_2, groups_y_2 = [G2, G4], [Y2, Y4]

        n_iters = 2 * len(G2)
        index_list = np.random.permutation(n_iters)

        n_iters_dcd = 4 * len(G2)
        index_list_dcd = np.random.permutation(n_iters_dcd)
       
        ###准备第一组训练的数据
        X1_G,X2_G = [],[]
        gt_y1,gt_y2,dcd_labels = [],[],[]

        for index_G in range(n_iters):
            ground_truth=index_list[index_G]//len(G2)
            x1, x2 = groups_2[ground_truth][index_list[index_G] - len(G2) * ground_truth]
            y1, y2 = groups_y_2[ground_truth][index_list[index_G] - len(G2) * ground_truth]
            
            dcd_label=0 if ground_truth==0 else 2
            X1_G.append(x1)  # 来自源域
            X2_G.append(x2)  # 来自目标域
            gt_y1.append(y1)
            gt_y2.append(y2)
            dcd_labels.append(dcd_label)    #同类则dcd=0，不同类dcd=2

            if (index_G+1)%config.mini_batch_size_g_h==0:  
                ###第一组训练的数据准备完成 
                #生成器的数据为：X1_G、X2_G、gt_y1、gt_y2、dcd_labels
                X1_G,X2_G = ops.stack(X1_G),ops.stack(X2_G)
                gt_y1,gt_y2 = ops.stack(gt_y1).astype(ms.int32),ops.stack(gt_y2).astype(ms.int32)
                dcd_labels = Tensor(dcd_labels,dtype=ms.int32)
                data1 = (X1_G,X2_G,gt_y1,gt_y2,dcd_labels)
                loss_ce = ce(data1)
                print('step3 Epoch {}/{}  iter-G {}/{} loss_ce: {:.4f}'.format(epoch + 1, config.n_epoch_3,index_G+1,n_iters,loss_ce.asnumpy().item()))
                #完成训练后，重置数据缓存区，进行下一次训练的数据准备
                X1_G,X2_G = [],[]
                gt_y1,gt_y2,dcd_labels = [],[],[]

        ###准备第二组训练的数据 
        X1_D,X2_D = [],[]
        gt_D = []

        for index_D in range(n_iters_dcd):
            ground_truth=index_list_dcd[index_D]//len(groups[1])
            x1, x2 = groups[ground_truth][index_list_dcd[index_D] - len(groups[1]) * ground_truth]
            X1_D.append(x1)
            X2_D.append(x2)
            gt_D.append(ground_truth)

            if (index_D + 1) % config.mini_batch_size_dcd == 0:
                ###第二组训练的数据准备完成            
                #鉴别器的数据为：X1_D、X2_D、gt_D

                X1_D,X2_D = ops.stack(X1_D),ops.stack(X2_D)
                gt_D = Tensor(gt_D,dtype=ms.int32)
                data2 = (X1_D,X2_D,gt_D)

                net_dd_loss = dd(data2)             
                print('step3 Epoch {}/{}  iter-D {}/{} loss_d: {:.4f}'.format(epoch + 1, config.n_epoch_3,index_D+1,n_iters_dcd,net_dd_loss.asnumpy().item()))
                #完成训练后，重置数据缓存区，进行下一次训练的数据准备
                X1_D,X2_D = [],[]
                gt_D = []

        ########接测试
        accuracy = eval_generator(encoder, classifier, test_dataloader)
        print("step3----Epoch %d/%d  accuracy: %.3f/%.3f " % (epoch + 1, config.n_epoch_3, accuracy,pre_acc))
        if accuracy>pre_acc:
            pre_acc = accuracy
            print('acc improved, save checkpoint...\n') 
            ms.save_checkpoint(encoder, os.path.join(current_dir, config.model_root, config.tgt_encoder_checkpoint))
            ms.save_checkpoint(classifier, os.path.join(current_dir, config.model_root, config.tgt_classifier_checkpoint))
            ms.save_checkpoint(discriminator, os.path.join(current_dir, config.model_root, config.tgt_discriminator_checkpoint))
            if accuracy>0.47:
                ms.save_checkpoint(encoder, os.path.join(current_dir, config.model_root, 'FADA-tgt-encoder47.ckpt'))
                ms.save_checkpoint(classifier, os.path.join(current_dir, config.model_root, "FADA-tgt-classifier47.ckpt"))
                ms.save_checkpoint(discriminator, os.path.join(current_dir, config.model_root, "FADA-tgt-discriminator47.ckpt"))
                    




class WithLossCellCE(nn.Cell):
    def __init__(self, classifier,encoder,discriminator,loss_fn):
        super(WithLossCellCE, self).__init__(auto_prefix=True)
        self.classifier = classifier
        self.encoder = encoder
        self.discriminator = discriminator
        self.loss_fn = loss_fn

    def construct(self, data1):
        X1,X2,label1,label2,label3 = data1
        encoder_X1,encoder_X2 = self.encoder(X1),self.encoder(X2)
        X_cat = ops.Concat(axis=1)([encoder_X1,encoder_X2])
        y_pred_X1,y_pred_X2=self.classifier(encoder_X1),self.classifier(encoder_X2)
        y_pred_dcd=self.discriminator(X_cat)

        loss_X1=self.loss_fn(y_pred_X1,label1)
        loss_X2=self.loss_fn(y_pred_X2,label2)
        loss_dcd=self.loss_fn(y_pred_dcd,label3)
        loss = loss_X1 + loss_X2 + 0.2 * loss_dcd
        return loss


class CE(nn.Cell):
    def __init__(self, train_for_CE):
        super(CE, self).__init__(auto_prefix=True)
        self.train_for_CE = train_for_CE

    def construct(self, data):
        out = self.train_for_CE(data)
        net_ce_loss = out.mean()
        return net_ce_loss
    

class DD(nn.Cell):
    def __init__(self, train_for_DD):
        super(DD, self).__init__(auto_prefix=True)
        self.train_for_DD = train_for_DD

    def construct(self, data):
        out = self.train_for_DD(data)
        net_dd_loss = out.mean()
        return net_dd_loss
    
class WithLossCellDD(nn.Cell):
    def __init__(self,encoder,discriminator,loss_fn):
        super(WithLossCellDD, self).__init__(auto_prefix=True)
        self.encoder = encoder
        self.discriminator = discriminator
        self.loss_fn = loss_fn

    def construct(self, data):
        X1,X2,label = data
        X_cat = ops.Concat(axis=1)([self.encoder(X1),self.encoder(X2)]).copy() 
        logits = self.discriminator(X_cat)
        loss = self.loss_fn(logits, label)
        return loss  
    

    