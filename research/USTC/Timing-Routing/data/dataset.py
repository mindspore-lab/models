# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/10/28 19:50
# @Author      : gwsun
# @Project     : RSMT
# @File        : dataset.py
# @env         : PyCharm
# @Description :
import os
from utils.myutil import get_length_batch
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.dataset as ds


class RandomRawdata(ds.GeneratorDataset):
    """
    最简单的数据集，只包含随机生成的数据坐标
    """
    def __init__(self, num, degree, file_path=None, use_coreset=True):  # num = batched * batch_size
        # self.arr, self.adj, self.adj_in, self.adj_out, self.mask = get_data(num, degree)
        if not use_coreset or file_path is None or not os.path.exists(file_path):
            print("data random generated.")
            cases = np.random.rand(num, degree, 2)
            unsampled_cases = np.round(cases, 8)
            self.arr = Tensor(unsampled_cases, mstype.float32)

        else:
            cases = np.load(file_path)
            cases = np.round(cases, 8)
            assert num == len(cases), 'Dataset size:{} is not match parameter num:{}!'.format(len(cases), num)
            assert degree == len(cases[0]), 'Degree is not match!'
            print('coreset read successfully, the size is {}'.format(num))
            self.arr = Tensor(cases, mstype.float32)

        super().__init__(source=self.generator_func(), column_names=["arr"], shuffle=False)

    def generator_func(self):
        for i in range(len(self.arr)):
            yield (self.arr[i],)

    def __len__(self):
        return len(self.arr)


class RandomRawdataEval(ds.GeneratorDataset):
    """
    最简单的数据集，只包含随机生成的数据坐标, 并且带有最优解长度
    """
    def __init__(self, num, degree, file_path=None):  # num = batched * batch_size
        # self.arr, self.adj, self.adj_in, self.adj_out, self.mask = get_data(num, degree)
        if file_path is None or not os.path.exists(file_path):
            cases = np.random.rand(num, degree, 2)
            unsampled_cases = np.round(cases, 8)
            self.arr = Tensor(unsampled_cases, mstype.float32)
            self.gst_lengths = get_length_batch(unsampled_cases)
            np.save('data/test_data/array_degree{}_num{}.npy'.format(degree, num), unsampled_cases)
            np.save('data/test_data/length_degree{}_num{}.npy'.format(degree, num), self.gst_lengths)
        else:
            unsampled_cases = np.load('data/test_data/array_degree{}_num{}.npy'.format(degree, num))
            self.arr = Tensor(unsampled_cases, mstype.float32)
            self.gst_lengths = np.load('data/test_data/length_degree{}_num{}.npy'.format(degree, num))

        super().__init__(source=self.generator_func(), column_names=["arr", "gst_lengths"], shuffle=False)

    def generator_func(self):
        for i in range(len(self.arr)):
            yield (self.arr[i], self.gst_lengths[i])

    def __len__(self):
        return len(self.arr)



