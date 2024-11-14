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
import moxing as mox
import time


parser = argparse.ArgumentParser(description='MindSpore Lenet Example')
parser.add_argument('--train_url',
                    help='output folder to save/load',
                    default=getRelatedPath('cache') + '/')

parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default=getRelatedPath('data') + '/')

parser.add_argument('--rank_size',
                    help='RANK_SIZE',
                    default='1')
args, unknown = parser.parse_known_args()

#多卡训练传入参数 rank_size
if os.getenv('RANK_SIZE') == None:
    os.putenv('RANK_SIZE', args.rank_size)
os.putenv('HCCL_CONNECT_TIMEOUT', '6000')
os.putenv('GLOG_v','1')
    

### Copy single dataset from obs to training image###
def ObsToEnv(obs_data_url, data_dir):
    try:     
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
    #Set a cache file to determine whether the data has been copied to obs. 
    #If this file exists during multi-card training, there is no need to copy the dataset multiple times.
    f = open("/cache/download_input.txt", 'w')    
    f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    except Exception as e:
        print("download_input failed")
    return 

### Copy the output to obs###
def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,obs_train_url) + str(e))
    return  

def UploadToQizhi(train_dir, obs_train_url):
    device_num = int(os.getenv('RANK_SIZE'))
    local_rank=int(os.getenv('DEVICE_ID'))
    if not os.path.exists(obs_train_url):
        os.mkdir(obs_train_url)
    if device_num == 1:
        EnvToObs(train_dir, obs_train_url)
    if device_num > 1:
        if local_rank%8==0:
            EnvToObs(train_dir, obs_train_url)
    return

def DownloadFromQizhi(obs_data_url, data_dir):
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        ObsToEnv(obs_data_url,data_dir)
    if device_num > 1:
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num = device_num, parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True, auto_parallel_search_mode="sharding_propagation")
        #Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
        local_rank=int(os.getenv('RANK_ID'))
        if local_rank%8==0:
            ObsToEnv(obs_data_url,data_dir)
        #If the cache file does not exist, it means that the copy data has not been completed,
        #and Wait for 0th card to finish copying data
        while not os.path.exists("/cache/download_input.txt"):
            time.sleep(1)  
    return


# set device_id and init for multi-card training
# set train_dir
train_dir = getRelatedPath('cache')
data_dir = getRelatedPath('raw_data')
device_num = int(os.getenv('RANK_SIZE'))
device_id=int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend', device_id=int(os.getenv('DEVICE_ID')))
DownloadFromQizhi(args.data_url, data_dir)
if device_num>1:
    init()
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num = device_num, parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True, auto_parallel_search_mode="sharding_propagation")

# task=None, model_name=None, dataset_name=None, config_file=None,saved_model=True, train=True, other_args=None
task = 'eta'  # 可选择的有：traffic_state_pred, traj_loc_pred, eta
model_name = 'DeepTTE'  # 填写model名字，可从model目录下查找
dataset_name = 'Beijing_Taxi_Sample'  # 填写对应的数据集名字
config_file = None  # 使用默认的配置文件
batch_size = 10



run_model(task=task, model_name=model_name, dataset_name=dataset_name, config_file=config_file, other_args={'rank_size':device_num,'batch_size':batch_size})

obs_dir='/cache/output'
###Copy the trained output data from the local running environment back to obs,
###and download it in the training task corresponding to the Qizhi platform
#This step is not required if UploadOutput is called
UploadToQizhi(args.train_url,obs_dir)