# Copyright 2024 Xidian University
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
from mindspore import nn
from mindspore import ops


class OffsetLoss(nn.Cell):
    def __init__(self):
        super(OffsetLoss, self).__init__()
    
    def construct(self, gt, pred):
        loss = ops.sum((gt - pred)**2, dim=1) / 100.
        return loss
