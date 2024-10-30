import os

from mindspore import context

from pipeline import run_model
from utils import getRelatedPath

os.environ["GLOG_v"] = "3"

train_dir = getRelatedPath('cache')
data_dir = getRelatedPath('raw_data')
context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', max_call_depth=20000, device_id=1)

task = 'traffic_state_pred'
model_name = 'DSTAGNN'
dataset_name = 'PEMSD8'
config_file = None

run_model(task=task, model_name=model_name, dataset_name=dataset_name, config_file=config_file)
