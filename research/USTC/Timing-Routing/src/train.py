import os
import argparse
import numpy as np
import torch
import sys
import time
from src.model import Actor, Critic
from utils.rsmt_utils import Evaluator
from utils.log_utils import *

from data.dataset import RandomRawdata, RandomRawdataEval
from utils.myutil import eval_len_from_adj, eval_distance

from mindspore.ops import operations as P

from src.init_env import init_env
from src.config import config
import mindspore
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore import nn
from mindspore.nn import optim
from mindspore import ops
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='train_loop', help='experiment name')
parser.add_argument('--degree', type=int, default=40, help='maximum degree of nets')
parser.add_argument('--batch_size', type=int, default=64, help='test batch size')
parser.add_argument('--eval_size', type=int, default=100, help='eval set size')
parser.add_argument('--num_batched', type=int, default=100, help='total number of the sample')
parser.add_argument('--seed', type=int, default=9, help='random seed')
parser.add_argument('--gpu_id', type=str, default='0,1')
parser.add_argument('--learning_rate', type=float, default=0.00003)
parser.add_argument("--weight", default=0.0, type=float, help='weight of radius in cost function.')
parser.add_argument("--sync_bn", default=-1)
args = parser.parse_args()

log_intvl = 100
radius_weight = args.weight
# 初始化分布式环境
init_env(config())

start_time = time.time()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
print('experiment', args.experiment)
base_dir = 'save/'
exp_dir = base_dir + args.experiment + '/'
log_dir = exp_dir + 'rsmt' + str(args.degree) + '.log'
ckp_dir = exp_dir + 'rsmt' + str(args.degree) + '.pt'
best_ckp_dir = exp_dir + 'rsmt' + str(args.degree) + 'b.pt'
pre_ckp_dir = '/home/dbcloud/sgw/RSMT/save/train_loop/rsmt35b.pt'
tensor_root = 'tensorboard_new/'

cur_dir = '_weight{}:{}_batchsize{}_degree'.format((1-radius_weight), radius_weight, args.batch_size)

best_eval = 100.
best_kept = 0

actor = Actor()
critic = Critic()
mse_loss = nn.MSELoss()
optimizer = optim.Adam(list(actor.trainable_params()) + list(critic.trainable_params()), args.learning_rate, weight_decay=1e-5)

start_batch = 1
batch_id = 0
evaluator = Evaluator()

train_dataset = RandomRawdata(args.num_batched * args.batch_size, args.degree,
                              file_path='data/coreset/sampled_kmeans_cases_degree{}.npy'.format(args.degree), use_coreset=False)
eval_dataset = RandomRawdataEval(args.eval_size, args.degree,
                                 file_path='data/test_data/length_degree{}_num{}.npy'.format(args.degree, args.eval_size))

train_loader = train_dataset.batch(args.batch_size)
eval_loader = eval_dataset.batch(args.batch_size)
grad_op = mindspore.ops.GradOperation(get_by_list=True)

for epoch in range(1):
    for data_sample in train_loader:
        arrs = data_sample[0]
        new_adj, log_probs, indexs = actor(arrs)
        predictions = critic(arrs, True)
        lengths = eval_len_from_adj(arrs, args.degree, new_adj)  # np.array
        radius = np.array(eval_distance(arrs, indexs, [0]*arrs.shape[0]))  # list
        length_tensor = Tensor((1-radius_weight) * lengths + radius_weight * radius)
        disadvantage = length_tensor - predictions
        actor_loss = ops.mean(disadvantage * log_probs)
        critic_loss = mse_loss(predictions, length_tensor)
        loss = actor_loss + critic_loss
        grads = grad_op(loss, list(actor.trainable_params()) + list(critic.trainable_params()))
        optimizer(grads)

