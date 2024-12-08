# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque

import mindspore


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0
    def entropy(self,p,is_prob=True,mean=True):
        if is_prob:
            p = np.softmax(p,axis=1)

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, mindspore.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: ({:.4f})".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Evaluation Metrics for Semantic Segmentation"""
import numpy as np
from mindspore.common.tensor import Tensor

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union','SimEntMetric']


class SimEntMetric():
    def __init__(self,):
        self.siment1_list=[]
        self.siment2_list=[]
        self.ent_list=[]

    def reset(self):
        self.siment1_list = []
        self.siment2_list = []
        self.ent_list = []

    def softmax(self,x, axis=1):
        """ softmax function """

        # assert(len(x.shape) > 1, "dimension must be larger than 1")
        # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行

        x -= np.max(x, axis=axis, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素

        # print("减去行最大值 ：\n", x)

        x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

        return x

    def entropy(self,p,is_prob=False,mean=True):
        if not is_prob:
            p=self.softmax(p)
        en = -np.sum(p*np.log(p+1e-9),1)
        if mean:
            return np.mean(en)
        else:
            return en



    def update(self,pred1, pred2):
        assert pred1.shape == pred2.shape
        pred1 = pred1.asnumpy()
        pred2 = pred2.asnumpy()
        self.ent_list.append(self.entropy(pred2))
        prob1=self.softmax(pred1)
        prob2=self.softmax(pred2)
        prob1 = prob1.transpose((0, 2, 3, 1))
        prob2 = prob2.transpose((0, 2, 3, 1))
        prob1 = prob1.reshape(-1,prob1.shape[3])
        prob2 = prob2.reshape(-1,prob2.shape[3])
        prob1_rand = np.random.permutation(prob1.shape[0])
        prob2_rand = np.random.permutation(prob2.shape[0])
        prob1 = np.linalg.norm(prob1[prob1_rand[:100]],axis=1,keepdims=True)
        prob2 = np.linalg.norm(prob2[prob2_rand[:100]],axis=1,keepdims=True)
        prob1_en =self.entropy(np.matmul(prob1,prob2.T)*20)
        prob2_en =self.entropy(np.matmul(prob2,prob1.T)*20)
        self.siment1_list.append(prob1_en)
        self.siment2_list.append(prob2_en)

    def get(self):
        return np.mean(self.siment1_list),np.mean(self.siment2_list),np.mean(self.ent_list)

class SegmentationMetric():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, ignore_label=255):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()
        self.ignore_label = ignore_label

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred.asnumpy(), label.asnumpy(), ignore_label=self.ignore_label)
            inter, union = batch_intersection_union(pred.asnumpy(), label.asnumpy(), self.nclass, ignore_label=self.ignore_label)

            self.total_correct += correct
            self.total_label += labeled
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)
        else:
            raise TypeError('The type of pred should equal to [Tensor | list[Tensor] | tuple[Tensor]], but now is {}'.format(type(preds)))

    def get(self, return_category_iou=False):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        # remove np.spacing(1)
        # print(self.total_correct, self.total_label)
        # print(self.total_inter, self.total_union)
        # raise None

        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        if return_category_iou:
            return pixAcc, mIoU, IoU
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.zeros(self.nclass)
        self.total_union = np.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0


def batch_pix_accuracy(output, target, ignore_label=255):
    """PixAcc"""
    # inputs are numpy array, output 4D NCHW where 'C' means label classes, target 3D NHW

    predict = np.argmax(output.astype(np.int64), 1)
    # predict = output.astype(np.int64)
    target = target.astype(np.int64)
    pixel_labeled = (target != ignore_label).sum()
    pixel_correct = ((predict == target) * (target != ignore_label)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass, ignore_label=255):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 0
    maxi = nclass - 1
    nbins = nclass
    predict = np.argmax(output.astype(np.float32), 1)
    # predict = output.astype(np.float32)
    target = target.astype(np.float32)
    predict[target == ignore_label] = ignore_label
    # predict = predict.astype(np.float32) * (target != ignore_label).astype(np.float32)
    # intersection = predict * (predict == target) * (target != ignore_label).astype(np.float32)
    intersection = predict[predict == target]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    # print(area_inter)
    # print(area_union)
    # print(len(predict),len(intersection), len(area_union))
    # print(np.unique(predict))
    # print(np.unique(target))
    assert (area_inter > area_union).sum() == 0, "Intersection area should be smaller than Union area"
    return area_inter.astype(np.float32), area_union.astype(np.float32)


# class SegmentationMetric(nn.Metric):
#     """FastSCNN Metric, computes pixAcc and mIoU metric scores."""
#     def __init__(self, nclass):
#         super(SegmentationMetric, self).__init__()
#         self.nclass = nclass
#         self.clear()
#
#     def clear(self):
#         """Resets the internal evaluation result to initial state."""
#         self.total_inter = np.zeros(self.nclass)
#         self.total_union = np.zeros(self.nclass)
#         self.total_correct = 0
#         self.total_label = 0
#
#     def update(self, *inputs):
#         """Updates the internal evaluation result.
#
#         Parameters
#         ----------
#         labels : 'NumpyArray' or list of `NumpyArray`
#             The labels of the data.
#         preds : 'NumpyArray' or list of `NumpyArray`
#             Predicted values.
#         """
#         preds, labels = inputs[0], inputs[-1]
#         preds = preds[0]
#         def evaluate_worker(self, pred, label):
#             correct, labeled = batch_pix_accuracy(pred.asnumpy(), label.asnumpy())
#             inter, union = batch_intersection_union(pred.asnumpy(), label.asnumpy(), self.nclass)
#
#             self.total_correct += correct
#             self.total_label += labeled
#             self.total_inter += inter
#             self.total_union += union
#
#         if isinstance(preds, Tensor):
#             evaluate_worker(self, preds, labels)
#         elif isinstance(preds, (list, tuple)):
#             for (pred, label) in zip(preds, labels):
#                 evaluate_worker(self, pred, label)
#
#     def eval(self):
#         """Gets the current evaluation result.
#
#         Returns
#         -------
#         metrics : tuple of float
#             pixAcc and mIoU
#         """
#         # remove np.spacing(1)
#         pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)
#         IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
#         mIoU = IoU.mean().item()
#         return pixAcc, mIoU

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target
