# Copyright 2020 Huawei Technologies Co., Ltd
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

from mindspore import nn, ops, Tensor, Parameter
import math


class AdaptiveAvgPool2D(nn.Cell):
    def __init__(self, size):
        super(AdaptiveAvgPool2D, self).__init__()
        if isinstance(size, int):
            size = [size, size]
        if isinstance(size, tuple):
            size = list(size)
        if not isinstance(size, list):
            raise ValueError('The size type should be int or tuple or list')

        self.size = size
        self.mean = ops.ReduceMean(keep_dims=True)
        self.slice = ops.Slice()

    @staticmethod
    def calculate(x, y):
        result_floor = x // y
        result_mod = x % y
        if result_mod == 0:
            result_ceil = result_floor
        else:
            result_ceil = result_floor + 1
        return result_floor, result_ceil

    def construct(self, inputs: Tensor):
        shape = inputs.shape
        h_out, w_out = self.size
        if h_out == None:
            h_out = shape[-2]
        if w_out == None:
            w_out = shape[-1]

        rate_1_f, rate_1_c = self.calculate(shape[-2], h_out)
        rate_2_f, rate_2_c = self.calculate(shape[-1], w_out)

        pooleds = []
        for i in range(h_out):
            pooleds_c = []
            for j in range(w_out):
                if inputs.ndim == 4:
                    start = (0, 0, i * rate_1_f, j * rate_2_f)
                    size = (shape[0], shape[1], rate_1_c, rate_2_c)
                else:
                    start = (0, i * rate_1_f, j * rate_2_f)
                    size = (shape[0], rate_1_c, rate_2_c)
                res = self.mean(self.slice(inputs, start, size), (inputs.ndim - 2, inputs.ndim - 1))
                pooleds_c.append(res)
            pooled_t = ops.Concat(-1)(pooleds_c)
            pooleds.append(pooled_t)
        pooleds = ops.Concat(-2)(pooleds)
        return pooleds


class AdaptiveMaxPool2D(nn.Cell):
    def __init__(self, size):
        super(AdaptiveMaxPool2D, self).__init__()
        if isinstance(size, int):
            size = [size, size]
        if isinstance(size, tuple):
            size = list(size)
        if not isinstance(size, list):
            raise ValueError('The size type should be int or tuple or list')

        self.size = size
        self.max = ops.ReduceMax(keep_dims=True)
        self.slice = ops.Slice()

    @staticmethod
    def calculate(x, y):
        result_floor = x // y
        result_mod = x % y
        if result_mod == 0:
            result_ceil = result_floor
        else:
            result_ceil = result_floor + 1
        return result_floor, result_ceil

    def construct(self, inputs: Tensor):
        shape = inputs.shape
        h_out, w_out = self.size
        if h_out == None:
            h_out = shape[-2]
        if w_out == None:
            w_out = shape[-1]

        rate_1_f, rate_1_c = self.calculate(shape[-2], h_out)
        rate_2_f, rate_2_c = self.calculate(shape[-1], w_out)

        pooleds = []
        for i in range(h_out):
            pooleds_c = []
            for j in range(w_out):
                if inputs.ndim == 4:
                    start = (0, 0, i * rate_1_f, j * rate_2_f)
                    size = (shape[0], shape[1], rate_1_c, rate_2_c)
                else:
                    start = (0, i * rate_1_f, j * rate_2_f)
                    size = (shape[0], rate_1_c, rate_2_c)

                temp = self.slice(inputs, start, size)
                res = self.max(temp, (inputs.ndim - 2, inputs.ndim - 1))
                pooleds_c.append(res)
            pooled_t = ops.Concat(-1)(pooleds_c)
            pooleds.append(pooled_t)
        pooleds = ops.Concat(-2)(pooleds)
        return pooleds


class AdaptiveMinPool2D(nn.Cell):
    def __init__(self, size):
        super(AdaptiveMinPool2D, self).__init__()
        if isinstance(size, int):
            size = [size, size]
        if isinstance(size, tuple):
            size = list(size)
        if not isinstance(size, list):
            raise ValueError('The size type should be int or tuple or list')

        self.size = size
        self.min = ops.ReduceMin(keep_dims=True)
        self.slice = ops.Slice()

    @staticmethod
    def calculate(x, y):
        result_floor = x // y
        result_mod = x % y
        if result_mod == 0:
            result_ceil = result_floor
        else:
            result_ceil = result_floor + 1
        return result_floor, result_ceil

    def construct(self, inputs: Tensor):
        shape = inputs.shape
        h_out, w_out = self.size
        if h_out == None:
            h_out = shape[-2]
        if w_out == None:
            w_out = shape[-1]

        rate_1_f, rate_1_c = self.calculate(shape[-2], h_out)
        rate_2_f, rate_2_c = self.calculate(shape[-1], w_out)

        pooleds = []
        for i in range(h_out):
            pooleds_c = []
            for j in range(w_out):
                if inputs.ndim == 4:
                    start = (0, 0, i * rate_1_f, j * rate_2_f)
                    size = (shape[0], shape[1], rate_1_c, rate_2_c)
                else:
                    start = (0, i * rate_1_f, j * rate_2_f)
                    size = (shape[0], rate_1_c, rate_2_c)

                temp = self.slice(inputs, start, size)
                res = self.min(temp, (inputs.ndim - 2, inputs.ndim - 1))
                pooleds_c.append(res)
            pooled_t = ops.Concat(-1)(pooleds_c)
            pooleds.append(pooled_t)
        pooleds = ops.Concat(-2)(pooleds)
        return pooleds


class AdaptiveSumPool2D(nn.Cell):
    def __init__(self, size):
        super(AdaptiveSumPool2D, self).__init__()
        if isinstance(size, int):
            size = [size, size]
        if isinstance(size, tuple):
            size = list(size)
        if not isinstance(size, list):
            raise ValueError('The size type should be int or tuple or list')

        self.size = size
        self.sum = ops.ReduceSum(keep_dims=True)
        self.slice = ops.Slice()

    @staticmethod
    def calculate(x, y):
        result_floor = x // y
        result_mod = x % y
        if result_mod == 0:
            result_ceil = result_floor
        else:
            result_ceil = result_floor + 1
        return result_floor, result_ceil

    def construct(self, inputs: Tensor):
        shape = inputs.shape
        h_out, w_out = self.size
        if h_out == None:
            h_out = shape[-2]
        if w_out == None:
            w_out = shape[-1]

        rate_1_f, rate_1_c = self.calculate(shape[-2], h_out)
        rate_2_f, rate_2_c = self.calculate(shape[-1], w_out)

        pooleds = []
        for i in range(h_out):
            pooleds_c = []
            for j in range(w_out):
                if inputs.ndim == 4:
                    start = (0, 0, i * rate_1_f, j * rate_2_f)
                    size = (shape[0], shape[1], rate_1_c, rate_2_c)
                else:
                    start = (0, i * rate_1_f, j * rate_2_f)
                    size = (shape[0], rate_1_c, rate_2_c)

                temp = self.slice(inputs, start, size)
                res = self.sum(temp, (inputs.ndim - 2, inputs.ndim - 1))
                pooleds_c.append(res)
            pooled_t = ops.Concat(-1)(pooleds_c)
            pooleds.append(pooled_t)
        pooleds = ops.Concat(-2)(pooleds)
        return pooleds


if __name__ == '__main__':
    import mindspore
    import numpy as np

    mindspore.set_context(mode=mindspore.GRAPH_MODE)
    input_x = Tensor(np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]), mindspore.float32)
    print(input_x.shape)
    adaptive_op = AdaptiveAvgPool2D((2, 2))
    output = adaptive_op(input_x)
    # print('output:', output.shape)
    print(output)
