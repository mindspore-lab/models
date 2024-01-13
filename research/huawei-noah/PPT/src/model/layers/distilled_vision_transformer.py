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

"""Implementation of DistilledVisionTransformer"""

import mindspore as ms
import mindspore.nn as nn

from mindspore import Parameter
from mindspore.ops import Zeros, Tile
from mindspore.ops.function import concat


from .weights_init import trunc_normal_
from .vision_transformer import VisionTransformer
from .custom_identity import CustomIdentity

class DistilledVisionTransformer(VisionTransformer):
    """
    Distilled Vision Transformer with support for
    patch or hybrid CNN input stage
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = Parameter(Zeros()((1, 1, self.embed_dim), ms.float32))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = Parameter(
            Zeros()((1, num_patches + 2, self.embed_dim), ms.float32)
        )
        self.head_dist = (
            nn.Dense(self.embed_dim, self.num_classes) if
            self.num_classes > 0 else CustomIdentity()
        )
        self.tile = Tile()
        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self.head_dist.cells(), self._init_weights)

    def forward_features(self, x):
        # taken from
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        b = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.tile(self.cls_token, (b, 1, 1))

        dist_token = self.tile(self.dist_token, (b, 1, 1))

        x = concat((cls_tokens, dist_token, x), axis=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        return (x + x_dist) / 2
