import os
import random
import re

import cv2
import mindspore
import numpy as np
import torch


# 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
from segment_anything import sam_model_registry


def pytorch_params(pth_file, verbose=False):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    # print(par_dict)
    if 'model' in par_dict and len(par_dict) < 10:
        par_dict = par_dict['model']
    for name, value in par_dict.items():
        if verbose:
            print(name, value.numpy().shape)
        pt_params[name] = value.numpy()
    return pt_params


# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network, verbose=False):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        if verbose:
            print(name, value.shape)
        ms_params[name] = value
    return ms_params


def mapper(ms_name: str):
    if 'gamma' in ms_name:
        return ms_name.replace('gamma', 'weight')
    if 'beta' in ms_name:
        return ms_name.replace('beta', 'bias')
    if 'embedding_table' in ms_name:
        return ms_name.replace('embedding_table', 'weight')
    return ms_name


def map_torch_to_mindspore(ms_dict, torch_dict, verbose=False):
    new_params_list = []
    for name, value in ms_dict.items():
        torch_name = mapper(name)
        torch_value = torch_dict[mapper(name)]

        convert_value = mindspore.Tensor(torch_value)

        if verbose:
            print(name, value.shape)
            # print(torch_name, value.shape)
        assert value.shape == convert_value.shape, f"value shape not match, ms {name} {value.shape}, torch {torch_name}{convert_value.shape}"
        new_params_list.append(dict(name=name, data=convert_value))
    return new_params_list


def convert_parameter(i_pth_path, i_ms_pth_path, ms_model):
    pt_param = pytorch_params(i_pth_path)
    print('\n'*2)
    ms_param = mindspore_params(ms_model)

    ms_params_list = map_torch_to_mindspore(ms_param, pt_param, verbose=False)

    print(f'successfully convert the checkpoint, saved as {i_ms_pth_path}')
    mindspore.save_checkpoint(ms_params_list, i_ms_pth_path)
    mindspore.load_checkpoint(i_ms_pth_path, ms_model)
    print(f'successfully load checkpoint into sam network')

    # compare
    print(f'torch image_encoder.blocks.4.norm1.bias', pt_param[f'image_encoder.blocks.4.norm1.bias'][:5])

    print(f'ms image_encoder.blocks.4.norm1.beta', ms_model.image_encoder.blocks[4].norm1.beta[:5])


if __name__ == "__main__":

    pth_path = os.path.join('./models', "sam_vit_l_0b3195.pth")
    ms_pth_path = os.path.join('./models', "ms-sam_vit_l.ckpt")

    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=None)

    convert_parameter(pth_path, ms_pth_path, ms_model=sam)
