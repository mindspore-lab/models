# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore import nn,ops
from . import Softmax

class WithLossCellG(nn.Cell):
    def __init__(self, lambda_, net_G, loss_fn1, size_source, batch_size=1,
                 num_classes=19):
        super(WithLossCellG, self).__init__(auto_prefix=True)
        self.lambda_ = lambda_
        self.net_G = net_G
        self.net_G.set_grad(True)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.size_source = size_source
        self.interp_source = ops.ResizeBilinear(size=(size_source[1], size_source[0]))
        self.loss_fn1 = loss_fn1
        self.softmax = Softmax(axis=1)
        self.zeros_like = ops.ZerosLike()

    def construct(self, image_source, label):
        pred1, pred2 = self.net_G(image_source)
        pred1 = self.interp_source(pred1)
        pred2 = self.interp_source(pred2)

        loss_seg1 = self.loss_fn1(pred1, label)
        loss_seg2 = self.loss_fn1(pred2, label)
        pred1 = self.softmax(pred1)
        pred2 = self.softmax(pred2)

        loss = loss_seg2 + self.lambda_[0] * loss_seg1

        loss_seg1 = ops.stop_gradient(loss_seg1)
        loss_seg2 = ops.stop_gradient(loss_seg2)

        pred1 = ops.stop_gradient(pred1)
        pred2 = ops.stop_gradient(pred2)

        return loss, (loss_seg1, loss_seg2), (pred1, pred2)


class TrainOneStepCellG(nn.Cell):
    def __init__(self, network, optimizer):
        super(TrainOneStepCellG, self).__init__(auto_prefix=False)
        self.network = network
        self.optimizer = optimizer
        self.weight = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        out = self.network(*inputs)
        grads = self.grad(self.network, self.weight)(*inputs)
        out = ops.functional.depend(out, self.optimizer(grads))
        return out