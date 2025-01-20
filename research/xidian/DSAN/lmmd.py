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
'''Lmmd'''
import numpy as np
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


class LMMD_loss(nn.Cell):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.shape[0]) + int(target.shape[0])
        total = ops.concat([source, target], axis=0)
        total0 = ops.broadcast_to(ops.expand_dims(total, 0),
                                  (int(total.shape[0]), int(total.shape[0]), int(total.shape[1])))
        total1 = ops.broadcast_to(ops.expand_dims(total, 1),
                                  (int(total.shape[0]), int(total.shape[0]), int(total.shape[1])))
        bandwidth = ops.ReduceSum()(((total0 - total1) ** 2).sum(2)) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = []
        for i in range(kernel_num):
            bandwidth_list.append(bandwidth ** (kernel_mul ** i))
        kernel_val = [ops.exp(-((total0 - total1) ** 2).sum(2) / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def construct(self, source, target, s_label, t_label, s_label_pre, weight):
        batch_size = source.shape[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size, class_num=self.class_num)
        weight_ss = Tensor.from_numpy(weight_ss)
        weight_tt = Tensor.from_numpy(weight_tt)
        weight_st = Tensor.from_numpy(weight_st)
        kernels = self.guassian_kernel(source, target,
                                       kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = ms.Tensor([0])
        if np.sum(ops.isnan(sum(kernels)).asnumpy()):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]
        loss += ops.ReduceSum()(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        loss_cls = nn.NLLLoss()
        loss_cls_value = loss_cls(ops.log_softmax(s_label_pre,axis=1), s_label)
        loss_final = weight * loss + loss_cls_value
        print('loss_lmmd:{},loss_cls:{},weight:{}'.format(loss, loss_cls_value.mean(), weight))
        return loss_final

    def convert_to_onehot(self, sca_label, class_num=31):
        sca_label = np.array(sca_label, dtype=int)
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=31):
        batch_size = s_label.shape[0]
        s_sca_label = s_label.asnumpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum
        t_sca_label = t_label.max(1)[1].asnumpy()
        t_sca_label = t_label.argmax(1).asnumpy()
        t_vec_label = t_label.asnumpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum
        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr
        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)
        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
