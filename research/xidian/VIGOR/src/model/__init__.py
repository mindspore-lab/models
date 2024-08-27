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
from .backbone import vgg16
from .SAFA import SAFA
from .utils.var_init import default_recurisive_init, KaimingNormal

import math

from mindspore import nn
from mindspore import ops
from mindspore.common import initializer as init


class VIGOR(nn.Cell):
    def __init__(self, 
                 grd_H=320, grd_W=640, 
                 sat_h=640, sat_W=640, 
                 phase="train"):
        super(VIGOR, self).__init__()
        self.grd = vgg16()
        self.sat = vgg16()

        self.grd_head = SAFA(sa_num=8, H=grd_H, W=grd_W)
        self.sat_head = SAFA(sa_num=8, H=sat_h, W=sat_W)

        self.delta_head = nn.SequentialCell([
            nn.Dense(4096*2, 512),
            nn.ReLU(),
            nn.Dense(512, 2),
        ])

        self.custom_init_weight()
    
    def construct(self, x_sat, x_sat_semi, x_grd):
        B = x_sat.shape[0]
        sat = ops.cat((x_sat, x_sat_semi), axis=0)
        sat_feat = self.sat_head(self.sat(sat))
        sat_feat, sat_semi_feat = ops.split(sat_feat, (B, B), axis=0)
        
        grd_feat = self.grd_head(self.grd(x_grd))

        both_feat = ops.cat((sat_feat, grd_feat), axis=-1)
        delta = self.delta_head(both_feat)
        
        return sat_feat, sat_semi_feat, grd_feat, delta
    
    def inference(self, img, mode='query'):
        if mode == 'query':
            return self.grd_head(self.grd(img))
        elif mode == 'ref':
            return self.sat_head(self.sat(img))

    def custom_init_weight(self):
        """
        Init the weight of Conv2d and Dense in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(
                    KaimingNormal(a=math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(
                    init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
