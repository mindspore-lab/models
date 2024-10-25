import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()
    def construct(self, x,y):
        out = self.mul(x, y)
        return out

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(sens_param=True)
        self.network = network
    def construct(self, x,y):
        gout = self.grad(self.network)(x,y)
        return gout
class GradSec(nn.Cell):
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation(sens_param=True)
        self.network = network
    def construct(self, x,y):
        gout = self.grad(self.network)(x,y)
        return gout

net=Net()
firstgrad = Grad(net) # first order
secondgrad = GradSec(firstgrad) # second order
x = Tensor([0.1, 0.2, 0.3], dtype=mstype.float32)
y = Tensor([0.1, 0.2, 0.3], dtype=mstype.float32)
output = secondgrad(x,y)
print(output)
