# Copyright 2024 Xidian University
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
import mindspore as ms
from mindspore import nn
from mindspore import ops


class SpatialAware(nn.Cell):
    def __init__(self, in_dim, num=8):
        super(SpatialAware, self).__init__()
        hidden_dim = in_dim // 2
        self.w1, self.b1 = self.init_weights_(in_dim, hidden_dim, num)
        self.w2, self.b2 = self.init_weights_(hidden_dim, in_dim, num)

    def init_weights_(self, din, dout, dnum):
        weight_shape = (din, dout, dnum)
        bias_shape = (1, dout, dnum)
        
        weight = ops.normal(weight_shape, mean=0.0, stddev=0.005)
        bias = 0.1 * ops.ones(bias_shape)
        
        weight = ms.Parameter(weight)
        bias = ms.Parameter(bias)
        return weight, bias

    def construct(self, input_feature):
        # b = input_feature.shape[0]
        # mask, _ = ops.max(input_feature, axis=1)
        # mask = mask.view(b, -1)
        # mask = ops.einsum('bi,ijd->bjd', mask, self.w1) + self.b1
        # res = ops.einsum('bjd,jid->bid', mask, self.w2) + self.b2
        
        mask, _ = ops.max(input_feature, axis=1)
        in_dim, hidden_dim, sa_num = self.w1.shape
        B, dim = mask.shape
        mask_w1 = ops.matmul(mask, self.w1.view(dim, -1))
        mask_w1 = mask_w1.view(B, hidden_dim, sa_num)
        mask_w1 += self.b1

        res = ops.ones((B, in_dim, sa_num))
        for i in range(sa_num):
            res[:, :, i] = ops.matmul(mask_w1[:, :, i], self.w2[:, :, i])
        res += self.b2
        return res


class SAFA(nn.Cell):
    def __init__(self, sa_num=8, H=320, W=640) -> None:
        super(SAFA, self).__init__()
        # based on down-sampling
        in_dim = (H // 32) * (W // 32)
        self.sa = SpatialAware(in_dim, sa_num)
    
    def L2Normlization(self, feat):
        return feat / ops.norm(feat, ord=2, dim=1, keepdim=True)

    def construct(self, feat):
        B, C, _, _ = feat.shape
        feat = feat.view(B, C, -1)
        w = self.sa(feat)
        feat = ops.matmul(feat, w).view(B, -1)
        feat = self.L2Normlization(feat)
        return feat
