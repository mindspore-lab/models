import os
import click
import numpy as np
import argparse
from pathlib import Path
from logzero import logger
from mindspore import Tensor
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.train import load_checkpoint, load_param_into_net
from model.mlp import ConvMLP
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore import context
import ast
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

parser = argparse.ArgumentParser(description='Train CMLPNet')

parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='train modelarts')
parser.add_argument('--is_distributed', type=ast.literal_eval, default=False, help="use 8 npus")
parser.add_argument('--device_id', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch_size', type=int, default=128)
parser.add_argument('--dataset_choose', type=str, default='cifar10', help="cifar10 or cifar100")
parser.add_argument('--device_target', type=str, default='Ascend')
parser.add_argument('--save_checkpoint_path',
                    type=str,
                    default="./ckpt",
                    help='if is test, must provide\
                    path where the trained ckpt file')
parser.add_argument('--format',
                    type=str,
                    default="MINDIR",
                    help='if is test, must provide\
                    path where the trained ckpt file')
args = parser.parse_args()
args = parser.parse_args()




if __name__ == '__main__':
    dataset_choose = args.dataset_choose
    image_height, image_width = 224, 224
    if dataset_choose == 'cifar10':
        class_num = 10
    elif dataset_choose == 'cifar100':
        class_num = 100
    args.save_checkpoint_path = './'
    mode = None
    if args.is_distributed:
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE', '1'))
        context.set_context(device_id=device_id)
        ms.set_auto_parallel_context(device_num=device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True)
        init()
        rank_id = get_rank()
        ckpt_save_dir = os.path.join(args.save_checkpoint_path, "ckpt_" + str(rank_id) + "/")
    else:
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)
        rank_id = 0
        device_num = 1
        ckpt_save_dir = os.path.join(args.save_checkpoint_path, './')
    # profiles = Profiler()
    network = ConvMLP(blocks=[2, 4, 2], dims=[128, 256, 512], mlp_ratios=[2, 2, 2],
                      classifier_head=True, channels=64, n_conv_blocks=2, num_classes=class_num)

    network.set_train(False)
    ms.load_checkpoint("checkpoint/cifa10_CMLP.ckpt", network)
    logger.info('Load Checkpoint Success......')

    inputs =  Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    ms.export(network, inputs, file_name='CMLP', file_format=args.format)
