import argparse
import numpy as np


import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
from ipdb import set_trace
from mindspore import load_checkpoint, load_param_into_net
# from model_mindspore.deeplab_multi import DeeplabMulti
# from model import  EncoderImagePrecomp,EncoderText
from mindspore import save_checkpoint, Tensor
import torch


"""
输入torch模型的参数名称
输出列表：元素是二元组（变换之后权重名称，原本的权重名称）
"""



def updata_torch_to_ms(static_dict_ms, 
                       static_dict_torch,
                       ms_weight_list, 
                       torch_qianzui,
                       ms_qianzui):
    ms_names = list(static_dict_ms.keys())
    torch_names = list(static_dict_torch.keys())
    for i,torch_key in enumerate(torch_names):
        new_name = torch_key.replace(torch_qianzui, ms_qianzui)
        if "embed.weight" in new_name:
            new_name = new_name.replace("embed.weight", "embed.embedding_table")
        # if new_name not in ms_names:
        #     print(new_name)
        assert new_name in ms_names , "error name"
        param = mindspore.Tensor(static_dict_torch[torch_key].numpy())
        name = new_name
        ms_weight_list.append({'name': new_name, 'data': param})
    return ms_weight_list
    
    


def pth2ckpt(model_path, save_path, image_path, text_path, criterion_path):
    #加载torch权重
    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    torch_static_dict = checkpoint['model']  #0图模型权重， 1为文本模型权重
    torch_qianzuis = ["module","module","module"]

    #加载ms权重
    ms_paths = [image_path,
                text_path,
                criterion_path]
    ms_qianzui = ["net_image","net_caption","criterion.sim"]

    ms_weight_list = []
    for i,ms_path in enumerate(ms_paths):
        torch_weight = torch_static_dict[i]
        ms_weight = load_checkpoint(ms_path)
        ms_weight_list = updata_torch_to_ms(ms_weight, 
                                            torch_weight,
                                            ms_weight_list, 
                                            torch_qianzuis[i], 
                                            ms_qianzui[i])

    mindspore.save_checkpoint(ms_weight_list, save_path)
    print("success")



