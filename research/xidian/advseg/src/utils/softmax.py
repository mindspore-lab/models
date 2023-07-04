# Copyright 2023 Xidian University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor, Parameter
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P

import numpy as np

context.set_context(mode=context.PYNATIVE_MODE)


def softmax(x, axis=1):
    """ softmax function """

    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行

    x -= np.max(x, axis=axis, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素

    # print("减去行最大值 ：\n", x)

    x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

    return x


class Softmax(nn.Cell):
    def __init__(self, axis=-1):
        super(Softmax, self).__init__()
        self.axis = axis
        self.max = P.ReduceMax(keep_dims=True)
        self.sum = P.ReduceSum(keep_dims=True)
        self.sub = P.Sub()
        self.exp = P.Exp()
        self.div = P.RealDiv()
        self.cast = P.Cast()

    def construct(self, x):
        x = self.cast(x, mstype.float32)
        x = self.sub(x, self.max(x, self.axis))
        x = self.div(self.exp(x), self.sum(self.exp(x), self.axis))
        return x


if __name__ == '__main__':
    x = np.random.randint(low=1, high=5, size=(2, 3))  # 生成一个2x3的矩阵，取值范围在1-5之间
    print("原始 ：\n", x)

    x_ = softmax(x.copy())
    print("变换后 ：\n", x_)
    softmax_ = Softmax(axis=1)
    x_tensor = Tensor(x, mstype.float32)
    print(x_tensor)
    x_out = softmax_(x_tensor)
    print(x_out)
    x_out1 = ops.Softmax(axis=1)(x_tensor)
    print(x_out1)
