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

import mindspore
import mindspore.nn as nn
from mindspore import ops
import mindspore.ops.functional as F
import time
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


class Softmax(nn.Cell):
    def __init__(self, axis=-1):
        super(Softmax, self).__init__()
        self.axis = axis
        self.max = P.ReduceMax(keep_dims=True)
        self.sum = P.ReduceSum(keep_dims=True)
        self.sub = P.Sub()
        self.exp = P.Exp()
        self.div = P.RealDiv()
        self.cast = P.Cast()

    def construct(self, x):
        x = self.cast(x, mstype.float32)
        x = self.sub(x, self.max(x, self.axis))
        x = self.div(self.exp(x), self.sum(self.exp(x), self.axis))
        return x

    
class WithLossCellG(nn.Cell):

    def __init__(self, cfg, netG, netDm, netDu, loss_calc, bce_loss, prob_2_entropy, source_label):
        super(WithLossCellG, self).__init__(auto_prefix=True)
        self.netG = netG
        self.loss_calc = loss_calc
        self.IS_MULTI_LEVEL = cfg.TRAIN.MULTI_LEVEL
        self.LAMBDA_SEG_MAIN = cfg.TRAIN.LAMBDA_SEG_MAIN
        self.LAMBDA_SEG_AUX = cfg.TRAIN.LAMBDA_SEG_AUX
        self.LAMBDA_ADV_MAIN = cfg.TRAIN.LAMBDA_ADV_MAIN
        self.LAMBDA_ADV_AUX = cfg.TRAIN.LAMBDA_ADV_AUX
        self.bce_loss = bce_loss
        self.source_label = source_label
        self.netDm = netDm
        self.netDu = netDu
        self.interp = nn.ResizeBilinear()
        self.prob_2_entropy = prob_2_entropy
        self.softmax = Softmax(axis=1)

    def construct(self, src_img, src_label, src_size, tgt_img, tgt_size):
        pred_src_aux, pred_src_main = self.netG(src_img)
        
        if self.IS_MULTI_LEVEL:
            pred_src_aux = self.interp(pred_src_aux, size=src_size, align_corners=True)
            loss_seg_src_aux = self.loss_calc(pred_src_aux, src_label)
        else:
            loss_seg_src_aux = 0
        pred_src_main = self.interp(pred_src_main, size=src_size, align_corners=True)
        loss_seg_src_main = self.loss_calc(pred_src_main, src_label)
        loss_seg = (self.LAMBDA_SEG_MAIN * loss_seg_src_main
                    + self.LAMBDA_SEG_AUX * loss_seg_src_aux)
 
        pred_trg_aux, pred_trg_main = self.netG(tgt_img)
        
        if self.IS_MULTI_LEVEL:
            pred_trg_aux = self.interp(pred_trg_aux, size=tgt_size, align_corners=True)
            d_out_aux = self.netDu(self.prob_2_entropy(self.softmax(pred_trg_aux)))
            loss_adv_trg_aux = self.bce_loss(d_out_aux, self.source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = self.interp(pred_trg_main, size=tgt_size, align_corners=True)
        d_out_main = self.netDm(self.prob_2_entropy(self.softmax(pred_trg_main)))
        
        loss_adv_trg_main = self.bce_loss(d_out_main, self.source_label)
        loss_adv = (self.LAMBDA_ADV_MAIN * loss_adv_trg_main
                    + self.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        pred_src_main = ops.stop_gradient(pred_src_main)
        pred_src_aux = ops.stop_gradient(pred_src_aux)
        pred_trg_main = ops.stop_gradient(pred_trg_main)
        pred_trg_aux = ops.stop_gradient(pred_trg_aux)
        
        return loss_seg, loss_adv, pred_src_main, pred_src_aux, pred_trg_main, pred_trg_aux


class WithLossCellD(nn.Cell):

    def __init__(self, cfg, netG, netDm, netDu, loss_calc, bce_loss, prob_2_entropy, source_label, target_label):
        super(WithLossCellD, self).__init__(auto_prefix=True)
        self.netG = netG
        self.loss_calc = loss_calc
        self.IS_MULTI_LEVEL = cfg.TRAIN.MULTI_LEVEL
        self.bce_loss = bce_loss
        self.source_label = source_label
        self.target_label = target_label
        self.netDm = netDm
        self.netDu = netDu
        self.prob_2_entropy = prob_2_entropy
        self.softmax = Softmax(axis=1)

    def construct(self, pred_src_main, pred_src_aux, pred_trg_main, pred_trg_aux):

        pred_src_main = ops.stop_gradient(pred_src_main)
        pred_src_aux = ops.stop_gradient(pred_src_aux)
        pred_trg_main = ops.stop_gradient(pred_trg_main)
        pred_trg_aux = ops.stop_gradient(pred_trg_aux)

        if self.IS_MULTI_LEVEL:
            d_out_aux = self.netDu(self.prob_2_entropy(self.softmax(pred_src_aux)))
            loss_d_aux_src = self.bce_loss(d_out_aux, self.source_label)
            loss_d_aux_src = loss_d_aux_src / 2
        else:
            loss_d_aux_src = 0
        d_out_main = self.netDm(self.prob_2_entropy(self.softmax(pred_src_main)))
        loss_d_main_src = self.bce_loss(d_out_main, self.source_label)
        loss_d_main_src = loss_d_main_src / 2

        # train with target
        if self.IS_MULTI_LEVEL:
            d_out_aux = self.netDu(self.prob_2_entropy(self.softmax(pred_trg_aux)))
            loss_d_aux_tgt = self.bce_loss(d_out_aux, self.target_label)
            loss_d_aux_tgt = loss_d_aux_tgt / 2
        else:
            loss_d_aux_tgt = 0
        d_out_main = self.netDm(self.prob_2_entropy(self.softmax(pred_trg_main)))
        loss_d_main_tgt = self.bce_loss(d_out_main, self.target_label)
        loss_d_main_tgt = loss_d_main_tgt / 2

        return loss_d_main_src, loss_d_aux_src, loss_d_main_tgt, loss_d_aux_tgt


class CustomTrainOneStepCellG(nn.Cell):

    def __init__(self, netTrainG, optimizer_f, optimizer_c):
        super(CustomTrainOneStepCellG, self).__init__(auto_prefix=True)

        self.netTrainG = netTrainG
        self.optimizer_f = optimizer_f
        self.optimizer_c = optimizer_c
        self.weights_f = self.optimizer_f.parameters
        self.weights_c = self.optimizer_c.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss_seg, loss_adv, pred_src_main, pred_src_aux, pred_trg_main, pred_trg_aux = self.netTrainG(*inputs)
        grads_f = self.grad(self.netTrainG, self.weights_f)(*inputs)
        grads_c = self.grad(self.netTrainG, self.weights_c)(*inputs)
        # self.optimizer_f(grads_f)
        # self.optimizer_c(grads_c)
        loss_adv = F.depend(loss_adv, self.optimizer_f(grads_f))

        loss_adv = F.depend(loss_adv, self.optimizer_c(grads_c))

        return loss_seg, loss_adv, pred_src_main, pred_src_aux, pred_trg_main, pred_trg_aux


class CustomTrainOneStepCellD(nn.Cell):

    def __init__(self, netTrainD, optimizer_d_main, optimizer_d_aux):
        super(CustomTrainOneStepCellD, self).__init__(auto_prefix=True)

        self.netTrainD = netTrainD
        self.optimizer_d_main = optimizer_d_main
        self.optimizer_d_aux = optimizer_d_aux
        self.weights_d_main = self.optimizer_d_main.parameters
        self.weights_d_aux = self.optimizer_d_aux.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss_d_main_src, loss_d_aux_src, loss_d_main_tgt, loss_d_aux_tgt = self.netTrainD(*inputs)
        
        grads_d_main = self.grad(self.netTrainD, self.weights_d_main)(*inputs)
        grads_d_aux = self.grad(self.netTrainD, self.weights_d_aux)(*inputs)
        # self.optimizer_d_main(grads_d_main)
        # self.optimizer_d_aux(grads_d_aux)
        loss_d_aux_tgt = F.depend(loss_d_aux_tgt, self.optimizer_d_main(grads_d_main))
        loss_d_aux_tgt = F.depend(loss_d_aux_tgt, self.optimizer_d_aux(grads_d_aux))
        
        
        return loss_d_main_src, loss_d_aux_src, loss_d_main_tgt, loss_d_aux_tgt

    
class WithEvalCellSrc(nn.Cell):

    def __init__(self, model):
        super(WithEvalCellSrc, self).__init__(auto_prefix=True)

        self.model = model
        self.interp = nn.ResizeBilinear()

    def construct(self, data, size):

        outputs = self.model(data)[1]

        outputs = self.interp(outputs, size=size, align_corners=True)

        return outputs