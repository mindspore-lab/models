import torch
# 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    pt_list=[]
    pt_list1 = []
    for name in par_dict:
        if "num_batches_tracked" in name:
            pt_list1.append(name)
        else:
            parameter = par_dict[name]
            pt_list.append(name)
            # print(name, parameter.numpy().shape)
            pt_params[name] = parameter.numpy()
    return pt_params,pt_list,pt_list1

# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    md_list=[]
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        md_list.append(name)
        # print(name, value.shape)
        ms_params[name] = value
    return ms_params,md_list

from model.mlp import ConvMLP
from mindspore import nn
model=ConvMLP(blocks=[2, 4, 2], dims=[128, 256, 512], mlp_ratios=[2, 2, 2],
                     classifier_head=True, channels=64, n_conv_blocks=2,num_classes=10)
pth_path = "checkpoint/cifa10_CMLP.pth"
pt_param,pt_list,pt_list1 = pytorch_params(pth_path)
# print("="*20)
ms_param,md_list = mindspore_params(model)
# print("="*20)

import mindspore as ms
def param_convert(ms_params, pt_params, ckpt_path,pt_list,md_list):
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight",
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var"}
    new_params_list = []
    for i in range(len(md_list)):
        ms_param=md_list[i]
        pt_param=pt_list[i]
        if "conv_stages" in ms_param or 'tokenizer' in ms_param:
            if "gamma" in ms_param or "beta" in ms_param or "moving_mean" in ms_param or "moving_variance" in ms_param:
                ms_param_item = ms_param.split(".")
                pt_param_item = ['module']+ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
                pt_param1 = ".".join(pt_param_item)
                if pt_param1 in pt_params and pt_params[pt_param1].shape == ms_params[ms_param].shape:
                    ms_value = pt_params[pt_param1]
                    new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
                else:
                    print(ms_param, "not match in pt_params case1")
            else:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
                if pt_params[pt_param].shape != ms_params[ms_param].shape:
                    print(ms_param, "not match in pt_params case1")
        else :

            ms_value = pt_params[pt_param]
            new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            if pt_params[pt_param].shape != ms_params[ms_param].shape:
                print(ms_param, "not match in pt_params case1")
    # 保存成MindSpore的checkpoint
    ms.save_checkpoint(new_params_list, ckpt_path)

ckpt_path = "checkpoint/cifa10_CMLP.ckpt"
param_convert(ms_param, pt_param, ckpt_path,pt_list,md_list)