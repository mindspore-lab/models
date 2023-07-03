import mindspore.nn as nn
import mindspore.ops as ops
import mindspore
from mindspore import context, Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P

from src.advnet import AdaptSegNet


class AllOptimizer(nn.Optimizer):
    def __init__(self, params, optimizer, optimizer_D1, optimizer_D2, learning_rate=0.1, weight_decay=0.0, loss_scale=1.0):
        super(AllOptimizer, self).__init__(learning_rate, params)
        self.parameters = params
        self.optimizer = optimizer
        self.optimizer_D1 = optimizer_D1
        self.optimizer_D2 = optimizer_D2

    def construct(self, gradients):
        optim_result_1 = self.optimizer(gradients)
        optim_result_2 = self.optimizer_D1(gradients)
        optim_result_3 = self.optimizer_D2(gradients)
        return optim_result_1 + optim_result_2 + optim_result_3


def get_optimizer(config, net: AdaptSegNet):
    learning_rate = nn.PolynomialDecayLR(learning_rate=config.learning_rate, end_learning_rate=1e-9, decay_steps=config.num_steps, power=config.power)
    optimizer = nn.SGD(net.model_G.trainable_params(), learning_rate=learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

    learning_rate_D1 = nn.PolynomialDecayLR(learning_rate=config.learning_rate_D, end_learning_rate=1e-9, decay_steps=config.num_steps, power=config.power)
    optimizer_D1 = nn.Adam(net.model_D1.trainable_params(), learning_rate=learning_rate_D1, beta1=0.9, beta2=0.99)

    learning_rate_D2 = nn.PolynomialDecayLR(learning_rate=config.learning_rate_D, end_learning_rate=1e-9, decay_steps=config.num_steps, power=config.power)
    optimizer_D2 = nn.Adam(net.model_D2.trainable_params(), learning_rate=learning_rate_D2, beta1=0.9, beta2=0.99)
    return AllOptimizer(net.trainable_params(), optimizer, optimizer_D1, optimizer_D2)
