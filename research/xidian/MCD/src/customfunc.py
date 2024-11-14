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
import mindspore as ms
from mindspore.nn import Cell
from mindspore.ops import Abs
from mindspore.ops import GradOperation
from mindspore.ops import ReduceMean
from mindspore.ops import Softmax


class DiscrepancyLoss(Cell):
    def __init__(self):
        super(DiscrepancyLoss, self).__init__(auto_prefix=True)
        self._abs = Abs()
        self._softmax = Softmax()
        self._reduce_mean = ReduceMean()

    def construct(self, out1, out2):
        return self._reduce_mean(self._abs(self._softmax(out1) - self._softmax(out2)))


class StepAWithLossCell(Cell):
    def __init__(self, net, loss_fn):
        super(StepAWithLossCell, self).__init__(auto_prefix=True)
        self.net = net
        self.loss_fn = loss_fn

    def construct(self, img, label):
        _, output1, output2 = self.net(img)
        loss = self.loss_fn(output1, label) + self.loss_fn(output2, label)
        return loss


class StepBWithLossCell(Cell):
    def __init__(self, net, loss_fn1, loss_fn2):
        super(StepBWithLossCell, self).__init__(auto_prefix=True)
        self.net = net
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2

    def construct(self, img_s, img_t, label):
        _, output_s1, output_s2 = self.net(img_s)
        loss_s = self.loss_fn1(output_s1, label) + self.loss_fn1(output_s2, label)
        _, output_t1, output_t2 = self.net(img_t)
        loss_dis = self.loss_fn2(output_t1, output_t2)
        loss = loss_s - loss_dis
        return loss


class StepCWithLossCell(Cell):
    def __init__(self, net, loss_fn):
        super(StepCWithLossCell, self).__init__(auto_prefix=True)
        self.net = net
        self.loss_fn = loss_fn

    def construct(self, img):
        _, output1, output2 = self.net(img)
        return self.loss_fn(output1, output2)


class TrainStepACell(Cell):
    def __init__(self, model, optimizer_G, optimizer_C1, optimizer_C2):
        super(TrainStepACell, self).__init__(auto_prefix=True)
        self.model = model
        self.G = model.net.G
        self.C1 = model.net.C1
        self.C2 = model.net.C2
        self.G.set_grad(True)
        self.C1.set_grad(True)
        self.C2.set_grad(True)
        self.G_weights = ms.ParameterTuple(self.G.trainable_params())
        self.C1_weights = ms.ParameterTuple(self.C1.trainable_params())
        self.C2_weights = ms.ParameterTuple(self.C2.trainable_params())
        self.optimizer_G = optimizer_G
        self.optimizer_C1 = optimizer_C1
        self.optimizer_C2 = optimizer_C2
        self.grad = GradOperation(get_by_list=True)

    def construct(self, img, label):
        loss = self.model(img, label)
        grads_G = self.grad(self.model, self.G_weights)(img, label)
        grads_C1 = self.grad(self.model, self.C1_weights)(img, label)
        grads_C2 = self.grad(self.model, self.C2_weights)(img, label)
        self.optimizer_G(grads_G)
        self.optimizer_C1(grads_C1)
        self.optimizer_C2(grads_C2)
        return loss


class TrainStepBCell(Cell):
    def __init__(self, model, optimizer_C1, optimizer_C2):
        super(TrainStepBCell, self).__init__(auto_prefix=True)
        self.model = model
        self.G = model.net.G
        self.C1 = model.net.C1
        self.C2 = model.net.C2
        self.G.set_grad(False)
        self.C1.set_grad(True)
        self.C2.set_grad(True)
        self.C1_weights = ms.ParameterTuple(self.C1.trainable_params())
        self.C2_weights = ms.ParameterTuple(self.C2.trainable_params())
        self.optimizer_C1 = optimizer_C1
        self.optimizer_C2 = optimizer_C2
        self.grad = GradOperation(get_by_list=True)

    def construct(self, img_s, img_t, label):
        loss = self.model(img_s, img_t, label)
        grads_C1 = self.grad(self.model, self.C1_weights)(img_s, img_t, label)
        grads_C2 = self.grad(self.model, self.C2_weights)(img_s, img_t, label)
        self.optimizer_C1(grads_C1)
        self.optimizer_C2(grads_C2)
        return loss


class TrainStepCCell(Cell):
    def __init__(self, model, optimizer_G):
        super(TrainStepCCell, self).__init__(auto_prefix=True)
        self.model = model
        self.G = model.net.G
        self.C1 = model.net.C1
        self.C2 = model.net.C2
        self.G.set_grad(True)
        self.C1.set_grad(False)
        self.C2.set_grad(False)
        self.G_weights = ms.ParameterTuple(self.G.trainable_params())
        self.optimizer_G = optimizer_G
        self.grad = GradOperation(get_by_list=True)

    def construct(self, img):
        loss = self.model(img)
        grads_G = self.grad(self.model, self.G_weights)(img)
        self.optimizer_G(grads_G)
        return loss


class TrainStep(Cell):
    def __init__(self, stepA, stepB, stepC, num_k):
        super(TrainStep, self).__init__()
        self.stepA = stepA
        self.stepB = stepB
        self.stepC = stepC
        assert (isinstance(num_k, ms.Tensor)), f'the num_k must be mindspore.Tensor, but got {type(num_k)}'
        self.num_k = num_k

    def construct(self, img_s, label_s, img_t, label_t):
        i = ms.Tensor(0)
        lossA = self.stepA(img_s, label_s)
        lossB = self.stepB(img_s, img_t, label_s)
        lossC = ms.Tensor(0, dtype=ms.float32)
        while i < self.num_k:
            lossC += self.stepC(img_t)
            i += 1
        lossC /= self.num_k

        return lossA, lossB, lossC


class WithEvalCell(Cell):
    def __init__(self, network, loss_fn):
        super(WithEvalCell, self).__init__(auto_prefix=True)
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, img_t, label_t):
        _, output1, output2 = self.network(img_t)
        output_ensemble = output1 + output2
        loss = self.loss_fn(output1, label_t)
        return output1, output2, output_ensemble, loss
