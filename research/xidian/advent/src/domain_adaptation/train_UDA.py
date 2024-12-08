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
import sys

import mindspore
from mindspore import nn

from src.nets.discriminator import get_fc_discriminator

from src.utils.func import bce_loss
from src.utils.loss import cross_entropy_2d
# from utils.loss import entropy_loss
from src.utils.func import prob_2_entropy

# from src.loss import loss as LOss
from src.utils import learning_rates
from src.nets.train_model import WithLossCellG, WithLossCellD, CustomTrainOneStepCellG, CustomTrainOneStepCellD
from src.domain_adaptation.eval_UDA import eval_model


def train_advent(model, trainloader, targetloader,testloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    num_classes = cfg.NUM_CLASSES

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)

    conv_params_fea = list(filter(lambda x: ('layer1' in x.name) or ('layer2' in x.name)
                                            or ('layer3' in x.name) or ('layer4' in x.name) or (x.name == 'conv1.weight'),
                                  model.trainable_params()))

    conv_params_cls = list(filter(lambda x: ('layer5' in x.name) or ('layer6' in x.name), model.trainable_params()))

    lr_iter_f = learning_rates.poly_lr(cfg.TRAIN.LEARNING_RATE, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.MAX_ITERS, end_lr=0.0,
                                       power=cfg.TRAIN.POWER)
    lr_iter_c = learning_rates.poly_lr(cfg.TRAIN.LEARNING_RATE * 10, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.MAX_ITERS, end_lr=0.0,
                                       power=cfg.TRAIN.POWER)

    lr_iter_d_max = learning_rates.poly_lr(cfg.TRAIN.LEARNING_RATE_D, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.MAX_ITERS, end_lr=0.0,
                                           power=cfg.TRAIN.POWER)

    lr_iter_d_aux = learning_rates.poly_lr(cfg.TRAIN.LEARNING_RATE_D, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.MAX_ITERS, end_lr=0.0,
                                           power=cfg.TRAIN.POWER)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer_fea = nn.SGD(conv_params_fea,
                          learning_rate=lr_iter_f,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    optimizer_cls = nn.SGD(conv_params_cls,
                           learning_rate=lr_iter_c,
                           momentum=cfg.TRAIN.MOMENTUM,
                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_aux = nn.Adam(d_aux.trainable_params(), learning_rate=lr_iter_d_aux,
                              beta1=0.9, beta2=0.99)
    optimizer_d_main = nn.Adam(d_main.trainable_params(), learning_rate=lr_iter_d_max,
                               beta1=0.9, beta2=0.99)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    # loss_calc = cross_entropy_2d(num_cls=cfg.NUM_CLASSES, ignore_label=255)
    loss_calc = nn.CrossEntropyLoss(ignore_index=255)
    bce_loss = nn.BCEWithLogitsLoss()

    netG_with_loss = WithLossCellG(cfg, model, d_main, d_aux,
                                   loss_calc, bce_loss,
                                   prob_2_entropy, source_label)

    netD_with_loss = WithLossCellD(cfg, model, d_main, d_aux,
                                   loss_calc, bce_loss,
                                   prob_2_entropy, source_label, target_label)

    train_netG = CustomTrainOneStepCellG(netG_with_loss, optimizer_fea, optimizer_cls)

    train_netD = CustomTrainOneStepCellD(netD_with_loss, optimizer_d_main, optimizer_d_aux)
    
    train_netG.set_train()
    train_netD.set_train()
    best_iou = -0.1
    
    for i, datas in enumerate(zip(trainloader.create_dict_iterator(), targetloader.create_dict_iterator())):
                
        data_src, data_trg = datas
        
        src_input = data_src['data']
        src_label = data_src['label']
        tgt_input = data_trg['data']
        src_size = (input_size_source[1], input_size_source[0])
        tgt_size = (input_size_target[1], input_size_target[0])
        
        loss_seg, loss_adv, pred_src_main, pred_src_aux, pred_trg_main, pred_trg_aux = train_netG(
                src_input, src_label, src_size, tgt_input, tgt_size)
        
        loss_d_main_src, loss_d_aux_src, loss_d_main_tgt, loss_d_aux_tgt = train_netD(pred_src_main, pred_src_aux, pred_trg_main, pred_trg_aux)
        
        # loss_seg, loss_adv = train_netG(src_input, src_label, src_size, tgt_input, tgt_size)
        
        # loss_d_main_src, loss_d_aux_src, loss_d_main_tgt, loss_d_aux_tgt = train_netD(src_input, src_label, src_size, tgt_input, tgt_size)
        
        current_losses = {'loss_seg': loss_seg,
                          'loss_adv': loss_adv,
                          'loss_d_main_src': loss_d_main_src,
                          'loss_d_aux_src': loss_d_aux_src,
                          'loss_d_main_tgt': loss_d_main_tgt,
                          'loss_d_aux_tgt': loss_d_aux_tgt}

        if (i+1) % 10 ==0:
            print_losses(current_losses, i+1)

        if (i+1) % cfg.TRAIN.SAVE_PRED_EVERY == 0 :
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)

            now_iou = eval_model(cfg,model,testloader)
            model.set_train()

            if best_iou<now_iou:
                if os.path.exists(os.path.join(cfg.TRAIN.SNAPSHOT_DIR,f'model_best_iou_{best_iou:.2f}.ckpt')):
                    os.remove(os.path.join(cfg.TRAIN.SNAPSHOT_DIR,f'model_best_iou_{best_iou:.2f}.ckpt'))
                best_iou=now_iou
                mindspore.save_checkpoint(model, os.path.join(cfg.TRAIN.SNAPSHOT_DIR,f'model_best_iou_{best_iou:.2f}.ckpt'))


        if i >= cfg.TRAIN.EARLY_STOP - 1:
            break

        sys.stdout.flush()

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.asnumpy()


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    print(f'iter = {i_iter} {full_string}')


def train_domain_adaptation(model, trainloader, targetloader,testloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'AdvEnt':
        train_advent(model, trainloader, targetloader,testloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
