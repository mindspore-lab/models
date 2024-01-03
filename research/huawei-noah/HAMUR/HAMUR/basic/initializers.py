"""
-*- coding: utf-8 -*-
@Time    : 9/12/2023 3:54 pm
@Author  : Xiaopeng Li
@File    : initializers.py

"""
from mindspore.common.initializer import Initializer
from mindspore import Tensor
import numpy as np


class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.

    Args:
        mean (float): the mean of the normal distribution
        std (float): the standard deviation of the normal distribution
    """

    def __init__(self, mean=0.0, std=1.0):
        super(RandomNormal, self).__init__()
        self.mean = mean
        self.std = std

    def _initialize(self, arr):
        data = np.random.normal(self.mean, self.std, arr.shape).astype(arr.dtype)
        Tensor(data, arr.dtype).copy_to(arr)

# Usage example
# embedding_layer = nn.Embedding(vocab_size, embed_dim, weight_init=RandomNormal(mean, std))
