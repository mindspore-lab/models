from turtle import forward
# import torch
import numpy as np
import time
# from torch import nn
# from torch.nn import MarginRankingLoss
# from torch.nn import functional as F
import mindspore
from mindspore import nn
from mindspore import ops

class CustomLoss(nn.Cell):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.sub = ops.Sub()
        self.square = ops.Square()
        self.reduce_mean = ops.ReduceMean()

    def construct(self, output, label):
        diff = self.sub(output, label)
        diff_square = self.square(diff)
        loss = self.reduce_mean(diff_square)
        return loss


class DRD_rel_BCELoss(nn.LossBase):
    def __init__(self, left_alpha, right_alpha, reduction='mean'):
        super(DRD_rel_BCELoss, self).__init__(reduction)
        self.left_alpha = left_alpha
        self.right_alpha = right_alpha

    def construct(self, score, posi_trust_score, nega_trust_score, target):
        optim_target = (target - nega_trust_score)/(posi_trust_score - nega_trust_score)
        optim_target = ops.clip_by_value(optim_target, 0, 1)
        adjust_tag_left = 0.5*ops.sign(optim_target - ops.sigmoid(score)) + 0.5
        adjust_tag_right = 0.5*ops.sign(ops.sigmoid(score) - optim_target) + 0.5
        xi = ops.exp(self.left_alpha*(optim_target - ops.sigmoid(score))*adjust_tag_left + self.right_alpha*(ops.sigmoid(score) - optim_target)*adjust_tag_right)

        return self.get_loss(xi*(-(target)*ops.log(ops.sigmoid(score)*posi_trust_score + (1 - ops.sigmoid(score))*nega_trust_score) 
                                - (1-(target))*ops.log(ops.sigmoid(score)*(1 - posi_trust_score) + (1 - ops.sigmoid(score))*(1 - nega_trust_score))))  
        

class DRD_bias_BCELoss(nn.LossBase):
    def __init__(self, beta, reduction='mean'):
        super(DRD_bias_BCELoss, self).__init__(reduction)
        self.beta = beta

    def construct(self, rel_p, posi_trust_score, nega_trust_score, target):
        return self.get_loss(-(target)*ops.log(rel_p*ops.sigmoid(posi_trust_score) + (1 - rel_p)*ops.sigmoid(nega_trust_score)) 
                                - (1-(target))*ops.log(rel_p*(1 - ops.sigmoid(posi_trust_score)) + (1 - rel_p)*(1 - ops.sigmoid(nega_trust_score))) - self.beta*ops.logsigmoid(posi_trust_score - nega_trust_score))



if __name__=="__main__":
    pass
