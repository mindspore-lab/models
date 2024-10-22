# Copyright 2022 Huawei Technologies Co., Ltd
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

"""
-*- coding: utf-8 -*-
@File  : main_ctr.py
"""

import os
import time
import argparse
import datetime

from sklearn.metrics import roc_auc_score, log_loss
import mindspore as ms
from mindspore import nn
from mindspore.dataset import GeneratorDataset

from models import KARWithLossCell, KAR
from dataset import MyDataset
from utils import load_parse_from_json, setup_seed, str2list


def evaluate(model, dataset):
    batch_num = dataset.get_dataset_size()
    batch_size = dataset.get_batch_size()
    print('eval batch num', batch_num, 'batch size', batch_size)
    eval_data = dataset.create_tuple_iterator()
    begin_time = time.time()
    pred_list, label_list = [], []

    for _ in range(batch_num):
        data = next(eval_data)
        # print(type(data))
        preds = model(data[0], data[1], data[2], data[3],
                      data[4], data[5], data[6], data[7])
        pred_list.extend(preds.asnumpy().tolist())
        label_list.extend(data[-1].asnumpy().tolist())

    eval_time = time.time() - begin_time
    auc = roc_auc_score(y_true=label_list, y_score=pred_list)
    logloss = log_loss(y_true=label_list, y_pred=pred_list)
    return auc, logloss, eval_time


def train(args):
    trainset = MyDataset(args.data_dir, 'train', max_hist_len=args.max_hist_len,
                         aug_prefix=args.aug_prefix)
    testset = MyDataset(args.data_dir, 'test', max_hist_len=args.max_hist_len,
                        aug_prefix=args.aug_prefix)
    test_dataset = GeneratorDataset(source=testset, column_names=['iid', 'aid', 'hist_iid',
                                                                  'hist_aid', 'hist_rating',
                                                                  'hist_len', 'itm_aug',
                                                                  'hist_aug', 'label'])
    test_dataset = test_dataset.batch(args.batch_size)
    train_dataset = GeneratorDataset(source=trainset, column_names=['iid', 'aid', 'hist_iid',
                                                                    'hist_aid', 'hist_rating',
                                                                    'hist_len', 'itm_aug',
                                                                    'hist_aug', 'label'],
                                     shuffle=True)
    train_dataset = train_dataset.batch(args.batch_size)

    net = KAR(args, trainset)
    loss = nn.BCELoss(reduction='mean')
    net_with_loss = KARWithLossCell(net, loss)
    opt = nn.Adam(net.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)
    model = ms.Model(network=net_with_loss, optimizer=opt)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'kar')

    best_auc = 0
    # training
    for epoch in range(args.epoch_num):
        begin_time = time.time()
        model.train(1, train_dataset)
        train_time = time.time() - begin_time
        eval_auc, eval_ll, eval_time = evaluate(net, test_dataset)
        print("EPOCH %d , train time: %.5f, test time: %.5f, auc: %.5f, "
              "logloss: %.5f" % (epoch, train_time, eval_time, eval_auc, eval_ll))

        if eval_auc > best_auc:
            best_auc = eval_auc
            ms.save_checkpoint(net, save_path)
            print('model save in', save_path)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/ml-1m/proc_data/')
    parser.add_argument('--save_dir', default='../model/ml-1m/')
    parser.add_argument('--setting_path', type=str, default='', help='setting dir')

    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--timestamp', type=str,
                        default=datetime.datetime.now().strftime("%Y%m%d%H%M"))

    parser.add_argument('--epoch_num', default=20, type=int,
                        help='epochs of each iteration.') #
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  #1e-3
    parser.add_argument('--weight_decay', default=0, type=float, help='l2 loss scale')  #0
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')  #0
    parser.add_argument('--convert_dropout', default=0.0, type=float,
                        help='dropout rate of convert module')  # 0
    parser.add_argument('--patience', default=3, type=int,
                        help='The patience for early stop')

    parser.add_argument('--augment', default='true', type=str,
                        help='whether to use augment vectors')
    parser.add_argument('--aug_prefix', default='bert_avg',
                        type=str, help='prefix of augment file')
    parser.add_argument('--max_hist_len', default=10, type=int,
                        help='the max length of user history')
    parser.add_argument('--embed_dim', default=32, type=int,
                        help='size of embedding')  # 32
    parser.add_argument('--final_mlp_arch', default='200,80', type=str2list,
                        help='size of final layer')
    parser.add_argument('--convert_arch', default='128,32', type=str2list,
                        help='size of convert net (MLP/expert net in MoE)')
    parser.add_argument('--expert_num', default=2, type=int,
                        help='number of expert')
    parser.add_argument('--specific_expert_num', default=6, type=int,
                        help='number of specific expert in PLE')
    parser.add_argument('-din_mlp', default='32,16', type=str2list,
                        help='MLP layer in DIN attention')

    args, _ = parser.parse_known_args()
    args.augment = True if args.augment.lower() == 'true' else False

    print('max hist len', args.max_hist_len)

    return args


if __name__ == '__main__':
    ARGS = parse_args()
    if ARGS.setting_path:
        ARGS = load_parse_from_json(ARGS, ARGS.setting_path)
    setup_seed(ARGS.seed)

    print('parameters', ARGS)
    train(ARGS)
