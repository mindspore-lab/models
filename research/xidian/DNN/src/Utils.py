import mindspore.nn as nn
import mindspore
import numpy.random
from mindspore import Tensor
import mindspore.ops as ops
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# resize image to size 32x32
cv2_scale36 = lambda x: cv2.resize(x, dsize=(36, 36),
                                   interpolation=cv2.INTER_LINEAR)
cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)
# reshape image
np_reshape = lambda x: np.reshape(x, (32, 32, 1))

class L2Norm(nn.Cell):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
        self.sqrt = ops.Sqrt()
        self.sum = ops.ReduceSum()

    def construct(self, x):
        norm = self.sqrt(self.sum(x*x, 1) + self.eps)
        x= x / ops.ExpandDims()(norm, -1).expand_as(x)
        return x

class L1Norm(nn.Cell):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
        self.sum = ops.ReduceSum()
        self.abs = ops.Abs()
    def construct(self, x):
        norm = self.sum(self.abs(x), axis = 1) + self.eps
        x= x / norm.expand_as(x)
        return x


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def orthogonal(shape, gain = 1):
    mul = ops.Mul()
    flat_shape = (shape[0], np.prod(shape[1:]))
    # import pdb
    # pdb.set_trace()
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = Tensor.from_numpy(q).astype(mindspore.float32)
    weight = mul(q.reshape(shape), gain)
    return weight

def eval_show(epoch_per_eval, path, args):
    plt.xlabel("epoch number")
    plt.ylabel("FPR95")
    plt.title("test_accuracy chart")
    color = ['orangered', 'blueviolet', 'green']
    for id ,key in enumerate(epoch_per_eval):
        plt.plot(epoch_per_eval[key]["epoch"], epoch_per_eval[key]["FPR95"], color=color[id], linestyle='-', label=key)
    plt.legend()
    plt.savefig(os.path.join(path, "train_epochs_" + str(args.epochs) + "_fpr95.png"))
    plt.show()

np.seterr(all='ignore')

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def sigmoid_tensor(x):
    return 1. / (1 + ops.Exp()(-x))

def sigmrnd(x):
    return ((1. / (1 + np.exp(-x))) > numpy.random.uniform(low=0,high=1,size=x.shape)).astype(numpy.float)

def sigmrnd_tensor(x):
    seed=numpy.random.randint(1,10000)

    return ((1. / (1 + ops.Exp()(-x))) > Tensor(numpy.random.uniform(low=0,high=1,size=x.shape),mindspore.float32)).astype(mindspore.float32)

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

def softmax_tensor(x):
    e = ops.Exp()(x - x.max())  # prevent overflow
    if e.ndim == 1:
        return e / ops.ReduceSum()(e, axis=0)
    else:
        return e / [ops.ReduceSum()(e, axis=1)].T  # ndim = 2