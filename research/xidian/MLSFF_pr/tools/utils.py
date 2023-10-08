#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:00:21 2018

@author: ws
"""



from ipdb import set_trace
import mindspore as ms
import mindspore.ops as ops
import mindspore.ops.operations as P
import numpy as np
import os
import json



expand_dims = ops.ExpandDims()
def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return ms.nn.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return ms.nn.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return ms.nn.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return ms.nn.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return ms.nn.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return ms.nn.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
    

def decode_sequence(ix_to_word, seq):
    out=[]
    for i_ind,i in enumerate(seq):
        txt = ''
        for j in range(len(i)):
            id_x = seq[i_ind,j]
            # set_trace()
            if id_x > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(id_x.item())]
            else:
                break
        out.append(txt)
    return out


class LanguageModelCriterion(ms.nn.Cell):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def construct(self, input, target, mask):
        # set_trace()
        target = target[:, :P.Shape()(input)[1]]
        mask =  mask[:, :P.Shape()(input)[1]]
        tar = expand_dims(target,2)
        tar = ops.Cast()(tar, ms.int64)
        # tar.set_dtype(ms.int64)
        # set_trace()
        output = -ops.GatherD()(input,2,tar) 
        output = ops.squeeze(output,2) 
        output = output* mask
        output = ops.ReduceSum()(output) / ops.ReduceSum()(mask)
        
        return output

def var_wrapper(x, volatile=False):
    if type(x) is dict:
        return {k: var_wrapper(v, volatile) for k,v in x.items()}
    if type(x) is list or type(x) is tuple:
        return [var_wrapper(v, volatile) for v in x]
    if isinstance(x, np.ndarray):
        x = ms.tensor.from_numpy(x)
    x = ms.Tensor(x)

    return x
