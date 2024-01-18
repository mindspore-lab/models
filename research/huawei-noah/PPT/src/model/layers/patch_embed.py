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

"""Patch embedding layer """

import mindspore.nn as nn
from mindspore.ops import Reshape, Transpose


class PatchEmbed(nn.Cell):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        self.img_size = img_size
        img_size = (img_size, img_size)
        p_size = (patch_size, patch_size)
        num_patches = (img_size[1] // p_size[1]) * (img_size[0] // p_size[0])

        self.img_size = img_size
        self.patch_size = p_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=p_size,
            stride=p_size, pad_mode='pad', padding=0, has_bias=True
        )


    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        b, _, h, w = x.shape
        assert h == self.img_size[0], f'nput image height ({h}) doesn\'t match model ({self.img_size[0]}).'
        assert w == self.img_size[1], f'Input image width ({w}) doesn\'t match model ({self.img_size[1]}).'
        x = self.proj(x)
        x = Reshape()(x, (b, self.embed_dim, self.num_patches))
        x = Transpose()(x, (0, 2, 1))
        return x
