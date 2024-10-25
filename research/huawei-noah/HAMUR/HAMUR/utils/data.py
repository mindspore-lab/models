"""
-*- coding: utf-8 -*-
@Time    : 9/12/2023 3:57 pm
@Author  : Xiaopeng Li
@File    : data.py

"""
import mindspore
import numpy as np
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset


def get_auto_embedding_dim(num_classes):
    """
    Calculate the dims of embedding vector according to number of classes in the category.
    emb_dim = [6 * (num_classes)^(1/4)]
    Reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)

    Args:
        num_classes: number of classes in the category

    Returns:
        the dims of embedding vector
    """
    return int(np.floor(6 * np.power(num_classes, 0.26)))


class Dataset_pre:
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}, Tensor(self.y[index], mindspore.float32)

    def __len__(self):
        return len(self.y)


def create_dataset(dataset, batch_size=8, num_parallel_workers=8, drop_remainder=False):
    input_data = GeneratorDataset(dataset, column_names=["data", "label"])
    input_data = input_data.batch(batch_size, drop_remainder, num_parallel_workers)
    return input_data
