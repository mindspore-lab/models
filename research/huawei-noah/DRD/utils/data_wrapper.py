# from torch.utils.data import DataLoader, Dataset
# import torch
import numpy as np
import mindspore

class Wrap_Dataset_point1:
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, sel_tag, ips_weight, target, feature, use_cuda=True, sparse_tag=False):
        # self.sel_tag = mindspore.Tensor(sel_tag, dtype=mindspore.float32)
        # self.ips_weight = mindspore.Tensor(ips_weight, dtype=mindspore.float32)
        # self.target = mindspore.Tensor(target, dtype=mindspore.float32)
        # self.fes = mindspore.Tensor(feature, dtype=mindspore.float32)
        self.sel_tag = np.array(sel_tag,dtype=np.float32)
        self.ips_weight = np.array(ips_weight, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.fes = np.array(feature, dtype=np.float32)
        # self.sel_tag = sel_tag
        # self.ips_weight = ips_weight
        # self.target = target
        # self.fes = feature


    def __getitem__(self, index):
        return self.fes[index], self.target[index], self.sel_tag[index], self.ips_weight[index]

    def __len__(self):
        return len(self.sel_tag) 



