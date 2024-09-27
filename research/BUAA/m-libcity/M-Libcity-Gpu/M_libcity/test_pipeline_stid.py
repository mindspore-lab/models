from pipeline import run_model
import mindspore
import os
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init

from utils import getRelatedPath
import time
os.environ["GLOG_v"]="3" 

# # set device_id and init for multi-card training
# # set train_dir
train_dir = getRelatedPath('cache')
data_dir = getRelatedPath('raw_data')
context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU',max_call_depth=20000, device_id=0)

# task=None, model_name=None, dataset_name=None, config_file=None,saved_model=True, train=True, other_args=None
task = 'traffic_state_pred'  # 可选择的有：traffic_state_pred, traj_loc_pred, eta
model_name = 'STID'  # 填写model名字，可从model目录下查找
dataset_name = 'PEMSD4'  # 填写对应的数据集名字
config_file = None  # 使用默认的配置文件
batch_size = 64

run_model(task=task, model_name=model_name, dataset_name=dataset_name,config_file=config_file)
