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
# This file has been derived from the https://github.com/huggingface/pytorch-image-models
# repository and modified.
# ============================================================================

"""Visual Transformer definition

"""
from functools import partial

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import Zeros
from mindspore import Parameter, Tensor
from mindspore.ops.function import concat
from mindspore.ops import Tile

from .patch_embed import PatchEmbed
from .block import Block, pp_block_adaptive
from .weights_init import trunc_normal_
from .custom_identity import CustomIdentity

class VisionTransformer(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=nn.LayerNorm,
                 act_mlp_layer=partial(nn.GELU, approximate=False),
                 threshold=7e-5, r=50, pp_loc_list=[3, 6, 9]
                 ):
        super().__init__()

        # PPT setting
        self.r = r
        self.OP_threshold = threshold
        self.pp_loc_list = pp_loc_list
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            raise NotImplementedError(
                'This Layer was not iimplementes because all models from deit does not use it'
            )
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = ms.Parameter(
            Zeros()((1, 1, embed_dim), ms.float32)
        )
        self.pos_embed = ms.Parameter(
            Zeros()((1, num_patches + 1, embed_dim), ms.float32)
        )
        self.tile = Tile()
        self.pos_drop = nn.Dropout(1.0 - drop_rate)

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.CellList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_mlp_layer
            )
            for i in range(depth)])
        block_list = []
        for i in range(depth):
            if i in self.pp_loc_list:
                block_list.append(pp_block_adaptive(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                    norm_layer=norm_layer, act_layer=act_mlp_layer
                ))
            else:
                block_list.append(Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                    norm_layer=norm_layer, act_layer=act_mlp_layer
                ))
        self.blocks = nn.CellList(block_list)
        self.norm = norm_layer((embed_dim,))

        # Classifier head
        self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else CustomIdentity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self.cells(), self._init_weights)

    def apply(self, layer, fn):
        for l_ in layer:
            if hasattr(l_, 'cells') and len(l_.cells()) >= 1:
                self.apply(l_.cells(), fn)
            else:
                fn(l_)
    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Dense) and m.bias is not None:
                constant_init = ms.common.initializer.Constant(value=0)
                constant_init(m.bias)


    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else CustomIdentity()

    def forward_features(self, x):
        """Forward features"""
        b = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.tile(self.cls_token, (b, 1, 1))
        x = concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        n = x.shape[1]
        size = Tensor(ms.ops.ones((b, n, 1), ms.float16))
        for blk in self.blocks:
            x, size = blk(x, size, self.r, self.OP_threshold)
        x = self.norm(x)
        return x[:, 0]

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = self.forward_features(x)
        x = self.head(x)
        return x
