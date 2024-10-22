import numpy as np
import mindspore
import mindspore as ms
import mindspore.nn as nn
import math
import pdb
import mindspore.ops as ops


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * mindspore.ops.Log(input_ + epsilon)
    entropy = mindspore.ops.ReduceSum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    # softmax_output = input_list[1].detach()
    softmax_output = input_list[1]
    feature = input_list[0]
    if random_layer is None:
        tmp1 = ops.expand_dims(softmax_output, 2)
        tmp2 = ops.expand_dims(feature, 1)
        batmatmul = ops.BatchMatMul()
        op_out = batmatmul(tmp1, tmp2)
        # op_out = mindspore.ops.BatchMatMul(x=tmp1, y=tmp2)
        ad_out = ad_net(op_out.view(-1, softmax_output.shape[1] * feature.shape[1]))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.shape[0] // 2
    tmp = np.array([[1]] * batch_size + [[0]] * batch_size)
    dc_target = mindspore.Tensor.from_numpy(tmp)
    dc_target = mindspore.Tensor(dc_target, dtype=ms.float32)
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+mindspore.ops.Exp(-entropy)
        source_mask = mindspore.ops.OnesLike(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = mindspore.ops.OnesLike(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / mindspore.ops.ReduceSum(source_weight).detach().item() + \
                 target_weight / mindspore.ops.ReduceSum(target_weight).detach().item()
        return mindspore.ops.ReduceSum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / mindspore.ops.ReduceSum(weight).detach().item()
    else:
        if ad_out.shape != dc_target.shape:
            print(1)
        return nn.BCELoss()(ad_out, dc_target).mean()

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = mindspore.tensor.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)
