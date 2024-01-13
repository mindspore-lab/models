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

import mindspore as ms


from src.model.layers.deit import (
    deit_base_patch16_224,
    deit_base_distilled_patch16_224,
    deit_tiny_patch16_224,
    deit_small_patch16_224,
    deit_tiny_distilled_patch16_224,
    deit_small_distilled_patch16_224,
    deit_base_patch16_384,
    deit_base_distilled_patch16_384
)

from src.model.layers.regnet import regnety_160

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
        'regnety_160': regnety_160
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


def create_teacher_model(
        model_name,
        checkpoint_path=None,
        **kwargs):
    """Create model by name with given parameters"""

    model = get_model_by_name(
        model_name, **kwargs
    )
    if checkpoint_path is not None:
        param_dict = ms.load_checkpoint(checkpoint_path)
        ms.load_param_into_net(model, param_dict)

    return model
