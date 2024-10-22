from pipeline import run_model
import mindspore
import os
import sys
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
import argparse
from utils import getRelatedPath


parser = argparse.ArgumentParser(description='MindSpore Lenet Example')
parser.add_argument('--train_url',
                    help='output folder to save/load',
                    default=getRelatedPath('cache') + '/')

parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default=getRelatedPath('data') + '/')

parser.add_argument('--model',
                    help='model name',default='STResNet')

parser.add_argument('--task',
                    help='task',default='traffic_state_pred')

parser.add_argument('--dataset',
                    help='dataset name',default='NYCBike20140409')

parser.add_argument('--rank_size',
                    help='RANK_SIZE',
                    default='1')
args, unknown = parser.parse_known_args()

#多卡训练传入参数 rank_size
if os.getenv('RANK_SIZE') == None:
    os.putenv('RANK_SIZE', args.rank_size)
os.putenv('HCCL_CONNECT_TIMEOUT', '6000')
os.putenv('GLOG_v','1')


# set device_id and init for multi-card training
# set train_dir
train_dir = getRelatedPath('cache')
data_dir = getRelatedPath('raw_data')
device_num = int(os.getenv('CUDA_NUM',1))
device_id=int(os.getenv('DEVICE_ID',0))
if device_num==1:
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', device_id=device_id)
else:
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    init("nccl")
    single_size = False
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num = device_num, parallel_mode=mindspore.ParallelMode.DATA_PARALLEL, gradients_mean=True,dataset_strategy="full_batch")

# task=None, model_name=None, dataset_name=None, config_file=None,saved_model=True, train=True, other_args=None
task = args.task  # 可选择的有：traffic_state_pred, traj_loc_pred, eta
model_name = args.model  # 填写model名字，可从model目录下查找
dataset_name = args.dataset # 填写对应的数据集名字
config_file = None  # 使用默认的配置文件
batch_size = 64

if int(args.rank_size) > 1:
    rank=get_rank()
    if rank != 0:
        output_file = open('log/out{}.log'.format(rank),'w')
        sys.stdout = output_file

run_model(task=task, model_name=model_name, dataset_name=dataset_name, config_file=config_file,other_args={'rank_size':device_num,'seed':13})

