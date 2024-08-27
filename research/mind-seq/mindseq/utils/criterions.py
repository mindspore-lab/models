# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mindspore as ms
import mindspore.nn as nn
import numpy as np

class QuantileLoss(nn.Cell):
    def __init__(self, config):
        super(QuantileLoss,self).__init__()
        self.q = ms.Tensor(config.quantiles)

    def construct(self, predictions, targets):
        diff = predictions - targets
        ql = (1-self.q)*ms.ops.relu(diff) + self.q*ms.ops.relu(-diff)
        losses = ql.view(-1, ql.shape[-1]).mean(0)
        return losses

def qrisk(pred, tgt, quantiles):
    diff = pred - tgt
    ql = (1-quantiles)*np.clip(diff,0, float('inf')) + quantiles*np.clip(-diff,0, float('inf'))
    losses = ql.reshape(-1, ql.shape[-1])
    normalizer = np.abs(tgt).mean()
    risk = 2 * losses / normalizer
    return risk.mean(0)
