# from torch.utils.data import DataLoader, Dataset
# import torch
# import numpy as np
import numpy as np
import mindspore

class Wrap_Dataset:
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, X, y):
        self.X = np.array(X, dtype=np.int64)
        self.y = np.array(y, dtype=np.float32)


    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y) 