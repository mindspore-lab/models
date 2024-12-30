import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)

class GradReverse(nn.Cell):
    def __init__(self, coeff):
        super(GradReverse, self).__init__()
        self.coeff = coeff
    def construct(self, x):
        x = x.asnumpy()
        return ms.Tensor(x)
    def bprop(self, x, out, grad_output):
        din = -self.coeff * grad_output
        return din

def adentropy(out, lamda, softmax):
    out = softmax(out)
    loss_adent = lamda * ops.mean(ops.reduce_sum(out * (ops.log(out + 1e-5)), 1))
    return loss_adent

def entropy(out, lamda, softmax):
    out = softmax(out)
    loss_ent = -lamda * ops.mean(ops.reduce_sum(out * (ops.log(out + 1e-5)), 1))
    return loss_ent

class Predictor(nn.Cell):
    def __init__(self, num_class=64, inc=4096, temp=0.05, grl_coeff=1):
        super(Predictor, self).__init__()
        self.fc = nn.Dense(inc, num_class, has_bias=False, weight_init='XavierUniform')
        self.num_class = num_class
        self.temp = temp
        self.grl = GradReverse(coeff=grl_coeff)

    def construct(self, x, reverse=False):
        if reverse:
            x = self.grl(x)
        x = x/(ops.norm(x, axis=1).view(-1, 1) + 1e-10)
        x_out = self.fc(x) / self.temp
        return x_out

class Predictor_deep(nn.Cell):
    def __init__(self, num_class=64, inc=4096, temp=0.05, grl_coeff=1):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Dense(inc, 512, weight_init='XavierUniform')
        self.fc2 = nn.Dense(512, num_class, has_bias=False, weight_init='XavierUniform')
        self.num_class = num_class
        self.temp = temp
        self.grl = GradReverse(coeff=grl_coeff)

    def construct(self, x, reverse=False):
        x = self.fc1(x)
        if reverse:
            x = self.grl(x)
        x = x/(ops.norm(x, axis=1).view(-1, 1) + 1e-10)
        x_out = self.fc2(x) / self.temp
        return x_out

class Discriminator(nn.Cell):
    def __init__(self, inc=4096, coeff=0.1):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Dense(inc, 512)
        self.fc2_1 = nn.Dense(512, 512)
        self.fc3_1 = nn.Dense(512, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.grl = GradReverse(coeff=coeff)

    def construct(self, x, reverse=False):
        if reverse:
            x = self.grl(x)
        x = self.relu1(self.fc1_1(x))
        x = self.relu2(self.fc2_1(x))
        x_out = self.sigmoid(self.fc3_1(x))
        return x_out

