# Copyright 2023 Xidian University
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

import mindspore
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.initializer import Normal


class ASPP_Classifier_V2(nn.Cell):
    def __init__(self, in_channels, dilation_series, padding_series, num_classes):
        super(ASPP_Classifier_V2, self).__init__()
        self.conv2d_list = nn.CellList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    in_channels,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    pad_mode='pad',
                    padding=padding,
                    dilation=dilation,
                    has_bias=True,
                    weight_init=Normal(0.01),
                )
            )

    def construct(self, x, size=None):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        if size is not None:
            out = P.ResizeBilinear(size, True)(out)
        return out