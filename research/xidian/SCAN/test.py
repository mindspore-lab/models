from vocab import Vocabulary
import src.evaluation as evaluation
from time import *
import os
from mindspore import context
import argparse
from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="data/", type=str,
                    help='data path')
parser.add_argument('--model_name', default="logs/checkpoint", type=str,
                    help='model path')
parser.add_argument('--num_epochs', default=20, type=int ,
                    help='num_epochs')
parser.add_argument('--device_target', default="Ascend", type=str,
                    help='Attention softmax temperature.')
parser.add_argument('--device_id', default=0, type=int ,
                    help='NPU id')
parser.add_argument('--best_val_weight_path', default="logs/checkpoint", type=str,
                    help='path of best val weight')
opt = parser.parse_args()

begin_time = time()
context.set_context(device_id=opt.device_id, mode=context.PYNATIVE_MODE, device_target=opt.device_target)
evaluation.test(model_path = opt.model_name, data_path=opt.data_path, split="test")
end_time = time()
run_time = end_time-begin_time
print ('该循环程序运行时间：',run_time) 