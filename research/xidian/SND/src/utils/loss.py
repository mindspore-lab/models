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

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore
from mindspore import context, Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P


# context.set_context(mode=context.PYNATIVE_MODE)


class SoftmaxCrossEntropyLoss(nn.Cell):
    def __init__(self, num_cls=19, ignore_label=255):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.greater_equal = P.GreaterEqual()
        self.logical_and = P.LogicalAnd()

    def construct(self, logits, labels):
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        # ops.Print()(self.greater_equal(labels_int, 0))
        # ops.Print()(self.not_equal(labels_int, self.ignore_label))
        weights = self.logical_and(self.greater_equal(labels_int, 0), self.not_equal(labels_int, self.ignore_label))
        weights = self.cast(weights, mstype.float32)
        # logits_ = self.cast(logits_, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss


class MaskedFill(nn.Cell):
    '''
    MaskedFill
    '''

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.select = ops.Select()
        self.fill = ops.Fill()
        self.cast = ops.Cast()

    def construct(self, inputs: Tensor, mask: Tensor):
        mask = self.cast(mask, mindspore.bool_)
        masked_value = self.fill(inputs.dtype, inputs.shape, self.value)
        output = self.select(mask, masked_value, inputs)
        return output


class CrossEntropy2d(nn.Cell):

    def __init__(self, ignore_label=255, mode='mean'):
        super(CrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.mode = mode
        self.cross_entropy = ops.SparseSoftmaxCrossEntropyWithLogits()
        self.tile = ops.Tile()
        self.masked_fill = MaskedFill(-1e-9)
        self.mask_select = ops.MaskedSelect()
        self.cast = ops.Cast()
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()
        self.logic_and = ops.LogicalAnd()
        self.logic_not = ops.LogicalNot()

    def data_mask(self, predict, target):
        n, c, h, w = self.shape(predict)
        # n, c, h, w = 5, 19, 16, 16
        # data1=ops.zeroslike()
        # target_mask=ops.Greater()(target,ops.ZerosLike()(target))
        # target_mask2=ops.NotEqual()(target,255)
        # target_mask = ops.LogicalAnd()(target_mask1,target_mask2)

        target_mask = self.logic_and(target >= 0, target != 255)
        target_mask = self.logic_not(target_mask)
        # target = self.mask_select(target, target_mask)

        target = self.masked_fill(target, target_mask)
        target = self.reshape(target, (-1,))
        target = self.cast(target, mindspore.int32)

        predict = self.transpose(predict, (0, 2, 3, 1))
        # predict_mask = target_mask.copy().view((n, h, w, 1))
        predict_mask = self.reshape(target_mask, (n, h, w, 1))

        predict_mask = self.cast(predict_mask, mindspore.int32)
        predict_mask = self.tile(predict_mask, (1, 1, 1, c))
        predict_mask = self.cast(predict_mask, mindspore.bool_)

        predict = self.masked_fill(predict, predict_mask)
        # predict = self.mask_select(predict, predict_mask)
        predict = self.reshape(predict, (-1, c))

        return predict, target

    def construct(self, predict, target, weight=None, mode='mean'):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        # print('2:', type(predict))

        # print(target.shape)
        # print(predict.shape)

        # loss = ops.SparseSoftmaxCrossEntropyWithLogits()(predict_, target)
        predict, target = self.data_mask(predict, target)
        # loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')(predict, target)
        loss = self.cross_entropy(predict, target).mean()
        # loss = .1
        return loss


def cross_entropy2d(predict, target, ignore_label=255, mode='mean'):
    n, c, h, w = predict.shape

    target_mask = ops.LogicalAnd()(target >= 0, target != ignore_label)
    target = ops.MaskedFill()(target, target_mask == False, 1e-9).view(-1).astype('int32')

    predict = predict.transpose(0, 2, 1, 3).transpose(0, 1, 3, 2)
    predict_mask = target_mask.reshape((n, h, w, 1))

    predict_mask = ops.Tile()(predict_mask, (1, 1, 1, c))
    predict = ops.MaskedFill()(predict, predict_mask == False, 1e-9).view((-1, c))
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction=mode)(predict, target.astype('int32'))

    return loss


class AllLoss(nn.Cell):
    def __init__(self, loss_fn1, loss_fn2, lambda_):
        super(AllLoss, self).__init__()
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.lambda_ = lambda_
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()

    def construct(self, output, label):
        (pred1, pred2, D_out1_part1, D_out2_part1), (out_s1_part2, out_s2_part2, out_t1_part2, out_t2_part2) = output
        loss_seg1 = self.loss_fn1(pred1, label)
        loss_seg2 = self.loss_fn1(pred2, label)
        loss_adv1_part1 = self.loss_fn2(D_out1_part1, self.zeros_like(D_out1_part1))
        loss_adv2_part1 = self.loss_fn2(D_out2_part1, self.zeros_like(D_out2_part1))
        loss_part1 = loss_seg2 + self.lambda_[0] * loss_seg1 + self.lambda_[2] * loss_adv2_part1 + self.lambda_[1] * loss_adv1_part1

        loss_part2 = (self.loss_fn2(out_s1_part2, self.zeros_like(out_s1_part2)) + self.loss_fn2(out_t1_part2, self.ones_like(out_t1_part2))) / 2.
        loss_part3 = (self.loss_fn2(out_s2_part2, self.zeros_like(out_s2_part2)) + self.loss_fn2(out_t2_part2, self.ones_like(out_t2_part2))) / 2.

        loss_display = (
            ops.stop_gradient(loss_seg2),
            ops.stop_gradient(loss_adv2_part1),
            ops.stop_gradient(loss_part3)
        )
        return loss_part1, loss_part2, loss_part3


def get_loss(config):
    calc_loss = SoftmaxCrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    return AllLoss(calc_loss, bce_loss, config.lambda_)


if __name__ == '__main__':
    import numpy as np

    np.random.seed(100)
    predict = np.random.random((5, 19, 16, 16))
    predict = mindspore.Tensor(predict, mindspore.float32)

    target = np.random.randint(0, 1, (5, 16, 16))
    target = mindspore.Tensor(target, mindspore.float32)

    # weight = mindspore.Tensor(np.ones((5, 16, 16),np.float32))

    # print(predict[0][0][0])
    print(type(predict), type(target))
    print(target.dtype, predict.dtype)

    # interp = nn.ResizeBilinear()
    # predict_new=interp(predict,size=(32,32))a
    # print('11:',predict_new.shape)

    # -------------------------------更换了一种写的方法，这种没有问题
    # loss = cross_entropy2d(predict, target)
    # print('loss:', loss)
    # -------------------------------Over

    # --------------------------------这部分代码测试显示有问题
    print('1:', type(predict))
    print(predict.dtype)
    loss = CrossEntropy2d()(predict, target)
    print(loss)
    loss = SoftmaxCrossEntropyLoss()(predict, target)
    print(loss)

    # ---------------------------------Over

    # nn.BCEWithLogitsLoss

    # from mindspore import Tensor
    #
    #
    # class Net(nn.Cell):
    #     def __init__(self):
    #         super(Net, self).__init__()
    #         self.binary_cross_entropy = ops.BinaryCrossEntropy()
    #
    #     def construct(self, logits, labels, weight):
    #         print('1', type(logits), type(labels))
    #         logits_=Tensor(logits)
    #         print(logits_.shape)
    #         # print(logits.mask_fill(logits>0.5),10)
    #         result = self.binary_cross_entropy(logits, labels, weight)
    #         return result
    #
    #
    # net = Net()
    # logits = Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float32)
    # labels = Tensor(np.array([0., 1., 0.]), mindspore.float32)
    # weight = Tensor(np.array([1, 2, 2]), mindspore.float32)
    # print(type(logits))
    # output = net(logits, labels, weight)
    # print(type(output))
