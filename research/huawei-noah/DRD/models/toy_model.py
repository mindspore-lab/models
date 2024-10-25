# import torch 
# import torch.nn as nn
# from torch.nn import Module
# from torch.nn import functional as F
# from torch.nn.init import xavier_normal_, constant_

import numpy as np
import math
from mindspore import nn
from mindspore.common.initializer import Normal,XavierNormal, Initializer, initializer,_calculate_fan_in_and_fan_out,_assignment

class XavierNormal(Initializer):
    def __init__(self, gain=1):
        super().__init__()
        # 配置初始化所需要的参数
        self.gain = gain

    def _initialize(self, arr): # arr为需要初始化的Tensor
        fan_in, fan_out = _calculate_fan_in_and_fan_out(arr.shape) # 计算fan_in, fan_out值

        std = self.gain * math.sqrt(2.0 / float(fan_in + fan_out)) # 根据公式计算std值
        data = np.random.normal(0, std, arr.shape) # 使用numpy构造初始化好的ndarray

        _assignment(arr, data) # 将初始化好的ndarray赋值到arr



class rank_model(nn.Cell):
    def __init__(self, in_size, hidden_size, drop_out):
        super(rank_model,self).__init__()
        self.linear_proj = nn.SequentialCell(
            # nn.Linear(in_size, 256),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            # nn.Linear(256, 128),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            # nn.Linear(128, 64),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            # nn.Linear(64, 1),
            nn.Dense(in_size, 256),
            nn.ELU(),
            nn.Dense(256, 128),
            nn.Dropout(p = drop_out),
            nn.ELU(),
            nn.Dense(128, 64),
            nn.Dropout(p = drop_out),
            nn.ELU(),
            nn.Dense(64, 1),
            # nn.Linear(in_size, 1, bias=True),
            )
        
    def weight_init(self):
        pass

    
    def construct(self, input_vec):
        # print('####################')
        # print(input_vec.shape)
        output = self.linear_proj(input_vec)
        # print('####################')
        return output


class bias_model(nn.Cell):
    def __init__(self, in_size, hidden_size, drop_out):
        super(bias_model,self).__init__()

        # self.linear_proj_posi = nn.SequentialCell(
        #     # nn.Linear(in_size, 1, bias=True),
        #     nn.Dense(in_size, 1, has_bias=False)
        #     # nn.Linear(in_size, 64),
        #     # nn.Linear(64, 32),
        #     # nn.Linear(32, 16),
        #     # nn.Linear(16, 1)
        #     )
        # self.linear_proj_nega = nn.SequentialCell(
        #     # nn.Linear(in_size, 1, bias=True),
        #     nn.Dense(in_size, 1, has_bias=False)
        #     # nn.Linear(in_size, 64),
        #     # nn.Linear(64, 32),
        #     # nn.Linear(32, 16),
        #     # nn.Linear(16, 1)
        #     # nn.Dense()
        #     )
        self.linear_proj_posi = nn.Dense(in_size, 1, has_bias=False)
        self.linear_proj_nega = nn.Dense(in_size, 1, has_bias=False)

    def weight_init(self):
         pass

    
    def construct(self, input_vec, rel_tag):
        output_consel = rel_tag * self.linear_proj_posi(input_vec) + (1- rel_tag) * self.linear_proj_nega(input_vec)

        return output_consel


class rank_model_linear(nn.Cell):
    def __init__(self, in_size, hidden_size, drop_out):
        super(rank_model_linear,self).__init__()
        # self.linear_proj = nn.Sequential(
        #         nn.Dense(in_size, 1, has_bias=True)
        #     )
        self.linear_proj = nn.Dense(in_size, 1, has_bias=True)
        
    def weight_init(self):
        pass

    
    def construct(self, input_vec):
        output = self.linear_proj(input_vec)
        return output


if __name__ =="__main__":
    model = rank_model(46, 16, 0.5)
    model.weight_init()
    print(model)
    for name, param in model.parameters_and_names():
        if 'weight' in name:
            param.set_data(initializer(XavierNormal(), param.shape, param.dtype))
        if 'bias' in name:
            param.set_data(initializer('zeros', param.shape, param.dtype))
    print('Param init OK')