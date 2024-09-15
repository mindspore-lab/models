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
from .soft_triplet import SoftTripletBiLoss
from .offset import OffsetLoss

from mindspore import nn
from mindspore import ops


class IoULoss(nn.Cell):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.softloss = SoftTripletBiLoss()
    
    def construct(self, sat_feat, sat_semi_feat, grd_feat, ratio):
        loss1, _, _ = self.softloss(sat_feat, grd_feat)
        # loss2 = self.softloss(sat_semi_feat, grd_feat)
        
        sim1 = ops.sum(sat_feat * grd_feat, dim=1)
        sim2 = ops.sum(sat_semi_feat * grd_feat, dim=1)
        
        error = (sim2/sim1) - ratio
        
        loss3 = ops.mean(error**2) / 10.
        # loss = loss1 + loss3
        
        # return loss, loss1, loss2, loss3
        return loss1, loss3

