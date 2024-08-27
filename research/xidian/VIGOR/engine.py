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
from mindspore.nn import Cell


class TrainStep(Cell):
    def __init__(self, model, loss_fn1, loss_fn2):
        super(TrainStep, self).__init__(auto_prefix=True)
        self.model = model
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
    
    def construct(self, sat, sat_semi, grd, gt_delta, ratio):
        sat_feat, sat_semi_feat, grd_feat, pred_delta = self.model(sat, sat_semi, grd)
        
        loss1, loss3 = self.loss_fn1(sat_feat, sat_semi_feat, grd_feat, ratio)
        loss2 = self.loss_fn2(gt_delta / 320., pred_delta)
        
        return loss1 + loss2 + loss3
