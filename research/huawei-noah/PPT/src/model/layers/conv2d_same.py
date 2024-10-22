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


from .padding import get_padding_value


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', 'valid')
    kwargs.setdefault('has_bias', False)
    padding, _ = get_padding_value(padding, kernel_size, **kwargs)
    if padding != 0:
        pad_mode = 'pad'
    else:
        pad_mode = 'valid'
    return nn.Conv2d(
        in_chs, out_chs, kernel_size,
        padding=padding, pad_mode=pad_mode, **kwargs
    )
