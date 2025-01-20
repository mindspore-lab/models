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
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import operations as P
from .var_init import default_recurisive_init

class PixelDiscriminator(nn.Cell):
    def __init__(self, input_nc, ndf=512, num_classes=1):
        super(PixelDiscriminator, self).__init__()

        self.D = nn.SequentialCell(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2))
        self.cls1 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, pad_mode='pad', padding=1)
        self.cls2 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, pad_mode='pad', padding=1)
        
        default_recurisive_init(self)
        
    def construct(self, x, size=None):
        out = self.D(x)
        src_out = self.cls1(out)
        tgt_out = self.cls2(out)
        out = ops.Concat(axis=1)((src_out, tgt_out))
        if size is not None:
            out = P.ResizeBilinear(size, True)(out)
        return out