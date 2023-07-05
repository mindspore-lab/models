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
"""Custom function for forward and evaluate."""
from mindspore.nn import Cell
from mindspore.nn import Softmax


class WithLossCell(Cell):
    def __init__(self, net, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=True)
        self._network = net
        self._loss_fn = loss_fn

    def construct(self, data1, data2, label):
        output = self._network((data1, data2))
        return self._loss_fn(output, label)


class WithEvalCell(Cell):
    def __init__(self, network):
        super(WithEvalCell, self).__init__(auto_prefix=True)
        self._network = network
        self._softmax = Softmax()

    def construct(self, data1, data2):
        outputs = self._network((data1, data2))
        return self._softmax(outputs)
