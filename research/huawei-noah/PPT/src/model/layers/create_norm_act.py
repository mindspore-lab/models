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
import types
import functools


from .norm_act import BatchNormAct2d


def convert_norm_act_type(norm_layer, act_layer, norm_kwargs=None):
    assert isinstance(norm_layer, (type, str, types.FunctionType, functools.partial))
    assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, functools.partial))
    norm_act_args = norm_kwargs.copy() if norm_kwargs else {}
    norm_act_layer = BatchNormAct2d
    # Must pass `act_layer` through for backwards compat where `act_layer=None` implies no activation.
    # In the future, may force use of `apply_act` with `act_layer` arg bound to relevant NormAct types
    # It is intended that functions/partial does not trigger this, they should define act.
    norm_act_args.update(dict(act_layer=act_layer))
    return norm_act_layer, norm_act_args
