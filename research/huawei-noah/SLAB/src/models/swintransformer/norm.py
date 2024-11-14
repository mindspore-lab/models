from mindspore import nn
from mindspore import Parameter
from mindspore import Tensor
import numpy as np
from mindspore import dtype as mstype


class RepBN(nn.Cell):
    def __init__(self, channels, epsilon=1e-5):
        super(RepBN, self).__init__()
        self.alpha = Parameter(Tensor(np.ones(1), dtype=mstype.float32))
        self.bn = nn.BatchNorm1d(channels[0])

    def construct(self, x):
        x = x.transpose(0, 2, 1)
        x = self.bn(x) + self.alpha * x
        x = x.transpose(0, 2, 1)
        return x


class LinearNorm(nn.Cell):
    def __init__(self, dim, norm1, norm2, epsilon=1e-5, warm=0, step=300000, r0=1.0):
        super(LinearNorm, self).__init__()
        self.warm = Parameter(Tensor(np.array(warm)), requires_grad=False)
        self.iter = Parameter(Tensor(np.array(step)), requires_grad=False)
        self.total_step = Parameter(Tensor(np.array(step)), requires_grad=False)
        self.r0 = r0
        self.norm1 = norm1(dim, epsilon=epsilon)
        self.norm2 = norm2(dim)

    def construct(self, x):
        if self.training:
            if self.warm > 0:
                self.warm = self.warm - 1
                x = self.norm1(x)
            else:
                lamda = self.r0 * self.iter.item() / self.total_step.item()
                if self.iter > 0:
                    self.iter = self.iter - 1
                x1 = self.norm1(x)
                x2 = self.norm2(x)
                x = lamda * x1 + (1 - lamda) * x2
        else:
            x = self.norm2(x)
        return x
