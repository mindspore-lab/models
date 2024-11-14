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


import mindspore.nn as nn



class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(BatchNormAct2d, self).__init__(
            num_features, eps=eps, momentum=1.0 - momentum, affine=affine)

        if act_layer is not None and apply_act:
            # act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer()
        else:
            self.act = None

    def _forward_python(self, x):
        return super(BatchNormAct2d, self).construct(x)

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = self._forward_python(x)
        if self.act is not None:
            x = self.act(x)
        return x
