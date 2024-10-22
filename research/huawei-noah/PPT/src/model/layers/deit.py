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

"""Define deit models"""


from functools import partial

import mindspore as ms
import mindspore.nn as nn

from src.model.layers.vision_transformer import VisionTransformer
from src.model.layers.distilled_vision_transformer import (
    DistilledVisionTransformer
)

from src.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



def _cfg(url='', **kwargs):
    return {
        'url': url, 'in_chans': 3,
        'num_classes': 1000, 'img_size': 224,
        'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


def deit_base_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = _cfg(img_size=224, **kwargs)
    return model


def deit_base_distilled_patch16_224(**kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        **kwargs)
    model.default_cfg = _cfg(img_size=224, **kwargs)

    return model


def deit_tiny_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = _cfg(img_size=224, **kwargs)
    return model


def deit_small_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = _cfg(img_size=224, **kwargs)
    return model


def deit_tiny_distilled_patch16_224(**kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        **kwargs)
    model.default_cfg = _cfg(img_size=224, **kwargs)
    return model


def deit_small_distilled_patch16_224(**kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        **kwargs)
    model.default_cfg = _cfg(img_size=224, **kwargs)
    return model


def deit_base_patch16_384(**kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = _cfg(img_size=384, **kwargs)

    return model


def deit_base_distilled_patch16_384(**kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = _cfg(img_size=384, **kwargs)
    return model


def get_model_by_name(model_name, **kwargs):
    """get network by name and initialize it"""

    models = {
        'deit_base_patch16_224': deit_base_patch16_224,
        'deit_base_distilled_patch16_224': deit_base_distilled_patch16_224,
        'deit_tiny_patch16_224': deit_tiny_patch16_224,
        'deit_small_patch16_224': deit_small_patch16_224,
        'deit_tiny_distilled_patch16_224': deit_tiny_distilled_patch16_224,
        'deit_small_distilled_patch16_224': deit_small_distilled_patch16_224,
        'deit_base_patch16_384': deit_base_patch16_384,
        'deit_base_distilled_patch16_384': deit_base_distilled_patch16_384,
    }
    return models[model_name](**kwargs)


def create_model(
        model_name,
        num_classes=1000,
        in_chans=3,
        checkpoint_path=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs):
    """Create model by name with given parameters"""

    model = get_model_by_name(
        model_name, num_classes=num_classes, in_chans=in_chans,
        drop_rate=drop_rate, drop_path_rate=drop_path_rate, **kwargs
    )
    if checkpoint_path is not None:
        param_dict = ms.load_checkpoint(checkpoint_path)
        ms.load_param_into_net(model, param_dict)

    return model
