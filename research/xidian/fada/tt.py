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

from mindspore.ops import stop_gradient
from mindspore import nn, Parameter, Tensor, ParameterTuple, ops
import numpy as np
import mindspore
from mindspore.ops import functional as F

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.w = Parameter(Tensor(np.array([6], np.float32)), name='w')
        self.b = Parameter(Tensor(np.array([1.0], np.float32)), name='b')

        self.w1 = Parameter(Tensor(np.array([6], np.float32)), name='w1')
        self.b1 = Parameter(Tensor(np.array([1.0], np.float32)), name='b1')

        self.w2 = Parameter(Tensor(np.array([6], np.float32)), name='w2')
        self.b2 = Parameter(Tensor(np.array([1.0], np.float32)), name='b2')

    def construct(self, x, y):
        out = x * self.w + self.b
        # 停止梯度更新，out对梯度计算无贡献

        out1 = out * self.w1 + self.b1

        # out1 = stop_gradient(out1)

        out2 = out * self.w2 + self.b2

        # return out
        return out1, out2


class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)

    def construct(self, x, y):
        out1, out2 = self.net(x, y)

        gradient_function = self.grad_op(self.net, self.params)

        out2 = F.depend(out2, gradient_function(x, y))
        return gradient_function(x, y), out1, out2

#         out1 = F.depend(out1, gradient_function(x, y))
    
x = Tensor([100], dtype=mindspore.float32)
y = Tensor([100], dtype=mindspore.float32)

output, out1, out2 = GradNet(Net())(x, y)

# for item in output:
#
#     output

print(output)
print(out1, out2)
print(f"wgrad: {output[0]}\nbgrad: {output[1]}")