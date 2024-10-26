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
from pathlib import Path
import time
import os.path as osp
import numpy as np

import mindspore
from mindspore import nn

from tqdm import tqdm

from model.discriminator import get_fc_discriminator
from utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from utils.func import bce_loss
from utils.loss import cross_entropy_2d
# from utils.loss import entropy_loss
from utils.func import prob_2_entropy
from utils.viz_segmask import colorize_mask

from src.loss import loss as LOss
from src.utils import learning_rates
from model.train_model import WithLossCellG, WithLossCellD, CustomTrainOneStepCellG, CustomTrainOneStepCellD


def train_advent(model, trainloader, targetloader, cfg):
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

    loss_calc = cross_entropy_2d(num_cls=cfg.NUM_CLASSES, ignore_label=255)

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

        print_losses(current_losses, i)

        if i % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)

            filename_f = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, "model_iter{:06d}_advent.ckpt".format(i))
            filename_d_aux = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, "model_iter{:06d}_d_main.ckpt".format(i))
            filename_d_main = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, "model_iter{:06d}_d_aux.ckpt".format(i))
            mindspore.save_checkpoint(model, filename_f)
            mindspore.save_checkpoint(d_aux, filename_d_aux)
            mindspore.save_checkpoint(d_main, filename_d_main)

        if i >= cfg.TRAIN.EARLY_STOP - 1:
            break

        sys.stdout.flush()


def train_minent(model, trainloader, targetloader, cfg):
    ''' UDA training with minEnt
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):

        # reset optimizers
        optimizer.zero_grad()

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training with minent
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main = model(images.cuda(device))
        pred_trg_aux = interp_target(pred_trg_aux)
        pred_trg_main = interp_target(pred_trg_main)
        pred_prob_trg_aux = F.softmax(pred_trg_aux)
        pred_prob_trg_main = F.softmax(pred_trg_main)

        loss_target_entp_aux = entropy_loss(pred_prob_trg_aux)
        loss_target_entp_main = entropy_loss(pred_prob_trg_main)
        loss = (cfg.TRAIN.LAMBDA_ENT_AUX * loss_target_entp_aux
                + cfg.TRAIN.LAMBDA_ENT_MAIN * loss_target_entp_main)
        loss.backward()
        optimizer.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_ent_aux': loss_target_entp_aux,
                          'loss_ent_main': loss_target_entp_main}

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


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


def train_domain_adaptation(model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'MinEnt':
        train_minent(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'AdvEnt':
        train_advent(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
