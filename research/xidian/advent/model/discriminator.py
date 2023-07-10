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

from mindspore import nn


def get_fc_discriminator(num_classes, ndf=64):
    return nn.SequentialCell(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, pad_mode='pad', padding=1),
        nn.LeakyReLU(alpha=0.2),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, pad_mode='pad', padding=1),
        nn.LeakyReLU(alpha=0.2),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, pad_mode='pad', padding=1),
        nn.LeakyReLU(alpha=0.2),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, pad_mode='pad', padding=1),
        nn.LeakyReLU(alpha=0.2),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, pad_mode='pad', padding=1),
    )
