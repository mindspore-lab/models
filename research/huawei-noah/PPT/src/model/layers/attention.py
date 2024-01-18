# Copyright 2023 Huawei Technologies Co., Ltd
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
#
# This file has been derived from the https://github.com/facebookresearch/deit
# repository and modified.
# ============================================================================


"""The implementation of the Attention layer for MindSpore framework."""

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import Reshape, Transpose, BatchMatMul, L2Normalize, Ones, Log
from mindspore.ops.function import concat


class Attention(nn.Cell):
    """
    The Attention layer
    The Pytorch implementation can be found by this link:
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L202
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1.0 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1.0 - proj_drop)

        self.reshape = Reshape()
        self.matmul = BatchMatMul()
        self.transpose = Transpose()
        self.softmax = nn.Softmax(axis=-1)
        self.ones = Ones()
        self.log = Log()

    def score_assignment_step(self, attn, v):
        B, H, _, _ = attn.shape
        C = v.shape[3] * H
        # breakpoint()
        v_norm = nn.Norm(axis=2)(self.reshape(self.transpose(v, (0, 2, 1, 3)), (B, attn.shape[2], C))) # value norm of size [B x N]
        significance_score = attn[:, :, 0].sum(axis=1)
        significance_score = significance_score * v_norm  # [B x N]
        significance_score = significance_score[:, 1:]  # [B x N-1]
        significance_score = significance_score / significance_score.sum(axis=1, keepdims=True)  # [B x N-1]
        
        return significance_score

    def construct(self, *inputs, **kwargs):
        x, size = inputs[0], inputs[1]
        b, n, c = x.shape
        qkv = self.reshape(
            self.qkv(x), (b, n, 3, self.num_heads, c // self.num_heads)
        )
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.matmul(q, self.transpose(k, (0, 1, 3, 2))) * self.scale

        if size is not None:
            attn = attn + self.log(size)[:, None, None, :, 0]
            
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        scores = self.score_assignment_step(attn, v) # [B, N-1]
        cls_score = 2 * self.ones((scores.shape[0], 1), scores.dtype) # confirm cls tokens are reserved
        scores = concat((cls_score, scores), axis=-1) # [B, N]

        x = self.transpose(self.matmul(attn, v), (0, 2, 1, 3))
        x = self.reshape(x, (b, n, c))

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, k.mean(axis=1), scores

