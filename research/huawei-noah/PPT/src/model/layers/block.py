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

"""The implementation of Block layer"""

from functools import partial
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from .custom_identity import CustomIdentity
from .attention import Attention
from .mlp import Mlp
from .drop_path import DropPath

from .tome import bipartite_soft_matching, merge_source, merge_wavg

# Copy and refine from DynamicViT
def batch_index_select(x, idx):
    if x is not None:
        if len(x.shape) == 3:
            B, N, C = x.shape
            N_new = idx.shape[1]
            offset = ops.arange(B).view(B, 1) * N
            idx = idx + offset
            out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
            return out
        elif len(x.shape) == 2:
            B, N = x.shape
            N_new = idx.shape[1]
            offset = ops.arange(B).view(B, 1) * N
            idx = idx + offset
            out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
            return out
        else:
            raise NotImplementedError
    else:
        return x

class Block(nn.Cell):
    """
    The Block layer
    The Pytorch implementation can be found by this link:
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L240
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=partial(nn.GELU, approximate=False),
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else CustomIdentity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def construct(self, *inputs, **kwargs):
        x, attn_size = inputs[0], inputs[1]
        x_attn, _, _ = self.attn(self.norm1(x), attn_size)
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_size

class pp_block_adaptive(nn.Cell):
    """
    The Block layer
    The Pytorch implementation can be found by this link:
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L240
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=partial(nn.GELU, approximate=False),
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else CustomIdentity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.sort = ops.Sort(axis=-1, descending=True)
        self.ones = ops.Ones()
        self.zeros = ops.Zeros()

    def construct(self, *inputs, **kwargs):
        x, attn_size, r, threshold = inputs[0], inputs[1], inputs[2], inputs[3]
        x_attn, metric, scores = self.attn(self.norm1(x), attn_size)
        x = x + self.drop_path(x_attn)
        
        x_pooling = None
        x_pruning = None
        attn_size_pooling = None
        attn_size_pruning = None
        
        r = min(r, (x.shape[1] - 1) // 2)
        S_op = scores[:, 1:].var(axis=1)
        # breakpoint()
        
        if r > 0:    
            # Apply pooling here
            pooling_imgs_indices = ops.nonzero(S_op < threshold)
            if pooling_imgs_indices.shape[0] > 0:
                pooling_imgs_indices = pooling_imgs_indices.squeeze(-1)
                x_pooling = x[pooling_imgs_indices]
                metric = metric[pooling_imgs_indices]
                attn_size_pooling = attn_size[pooling_imgs_indices]
                 # breakpoint()
                merge, _ = bipartite_soft_matching(metric, r)
                # breakpoint()
                x_pooling, attn_size_pooling = merge_wavg(merge, x_pooling, attn_size_pooling)

            # Apply pruning here
            pruning_imgs_indices = ops.nonzero(S_op >= threshold)
            if pruning_imgs_indices.shape[0] > 0:
                pruning_imgs_indices = pruning_imgs_indices.squeeze(-1)
                x_pruning = x[pruning_imgs_indices]
                scores = scores[pruning_imgs_indices]
                attn_size_pruning = attn_size[pruning_imgs_indices]
                _, sorted_indices = self.sort(scores)
                pruning_indices = sorted_indices[:, :-r] # reserved tokens
                x_pruning = batch_index_select(x_pruning, pruning_indices)
                attn_size_pruning = batch_index_select(attn_size_pruning, pruning_indices)

            # Merge two subbatch
            if x_pooling is not None and x_pruning is not None:
                x = self.zeros((x_pooling.shape[0]+x_pruning.shape[0], x_pooling.shape[1], x_pooling.shape[2]), x_pooling.dtype)
                x[pooling_imgs_indices], x[pruning_imgs_indices] = x_pooling, x_pruning
                attn_size = self.ones((x.shape[0], x.shape[1], 1), attn_size_pooling.dtype)
                attn_size[pooling_imgs_indices], attn_size[pruning_imgs_indices] = attn_size_pooling, attn_size_pruning
            elif x_pooling is not None:
                x = x_pooling
                attn_size = attn_size_pooling
            elif x_pruning is not None:
                x = x_pruning
                attn_size = attn_size_pruning
            else:
                pass        
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_size