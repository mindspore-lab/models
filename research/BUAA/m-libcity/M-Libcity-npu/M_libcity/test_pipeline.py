import os
os.environ['GLOG_v']='3'
os.environ['HCCL_CONNECT_TIMEOUT']='6000'
# os.environ['RANK_SIZE']='1'
from pipeline import run_model
import mindspore
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init
import shutil
import argparse
from utils import getRelatedPath
import time

parser = argparse.ArgumentParser(description='M_libcity config')
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
device_num = int(os.getenv('RANK_SIZE',1))
device_id=int(os.getenv('DEVICE_ID',0))
context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend', device_id=device_id)

if device_num>1:
    init()
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num = device_num, parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True, auto_parallel_search_mode="sharding_propagation")

# task=None, model_name=None, dataset_name=None, config_file=None,saved_model=True, train=True, other_args=None
task = args.task  # 可选择的有：traffic_state_pred, traj_loc_pred, eta
model_name = args.model  # 填写model名字，可从model目录下查找
dataset_name = args.dataset  # 填写对应的数据集名字
config_file = None  # 使用默认的配置文件
batch_size = 10


run_model(task=task, model_name=model_name, dataset_name=dataset_name, config_file=config_file, other_args={'rank_size':device_num})
