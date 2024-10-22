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


from mindspore import Tensor

class TaskLoss:
    def __init__(self, loss):
        self.loss = loss

    def backward(self):
        if self.loss.requires_grad:
            self.loss.backward()

    def get_loss_dict(self) -> dict:
        return getattr(self, '__dict__')


class LossDepot:
    def __init__(self):
        self.depot = dict()

    def add(self, loss):
        loss_dict = loss.get_loss_dict()

        for loss_name, loss_value in loss_dict.items():
            if loss_name not in self.depot:
                self.depot[loss_name] = []
            self.depot[loss_name].append(loss_value.detach().cpu().item())

    def summarize(self):
        for loss_name in self.depot:
            self.depot[loss_name] = Tensor(self.depot[loss_name]).mean().item()
        return self
