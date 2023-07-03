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


import sys
import os
from src.advnet.deeplabv2 import get_deeplab_v2

sys.path.append(f"{os.getcwd()}/src/advnet")

from mindspore import ops, ParameterTuple
from mindspore.ops import functional as F
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode, _get_enable_parallel_optimizer)
from mindspore.nn.wrap import DistributedGradReducer
from mindspore.context import ParallelMode

from src.model_utils import Softmax
from .deeplab import *
from .deeplab_multi import *
from .deeplabv2 import *
from .discriminator import *


# class Softmax(nn.Cell):
#     def __init__(self, axis=-1):
#         super(Softmax, self).__init__()
#         self.axis = axis
#         self.max = P.ReduceMax(keep_dims=True)
#         self.sum = P.ReduceSum(keep_dims=True)
#         self.sub = P.Sub()
#         self.exp = P.Exp()
#         self.div = P.RealDiv()
#         self.cast = P.Cast()
#
#     def construct(self, x):
#         x = self.cast(x, mstype.float32)
#         x = self.sub(x, self.max(x, self.axis))
#         x = self.div(self.exp(x), self.sum(self.exp(x), self.axis))
#         return x


class WithLossCellG(nn.Cell):
    def __init__(self, lambda_, net_G, net_D1, net_D2, loss_fn1, loss_fn2, size_source, size_target, batch_size=1,
                 num_classes=19):
        super(WithLossCellG, self).__init__(auto_prefix=True)
        self.lambda_ = lambda_
        self.net_G = net_G
        self.net_D1 = net_D1
        self.net_D2 = net_D2
        self.net_G.set_grad(True)
        self.net_D1.set_grad(False)
        self.net_D2.set_grad(False)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.size_source = size_source
        self.size_target = size_target
        self.interp_source = ops.ResizeBilinear(size=(size_source[1], size_source[0]))
        self.interp_target = ops.ResizeBilinear(size=(size_target[1], size_target[0]))
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.softmax = Softmax(axis=1)
        self.zeros_like = ops.ZerosLike()

    def construct(self, image_source, label, image_target):
        # time_1 = time.time()
        pred1, pred2 = self.net_G(image_source)
        pred1 = self.interp_source(pred1)
        pred2 = self.interp_source(pred2)

        loss_seg1 = self.loss_fn1(pred1, label)
        loss_seg2 = self.loss_fn1(pred2, label)
        pred1_target, pred2_target = self.net_G(image_target)

        pred1_target = self.interp_target(pred1_target)
        pred2_target = self.interp_target(pred2_target)
        pred1_target = self.softmax(pred1_target)
        pred2_target = self.softmax(pred2_target)

        out_D1 = self.net_D1(pred1_target)
        out_D2 = self.net_D2(pred2_target)
        source_label1 = self.zeros_like(out_D1)
        source_label2 = self.zeros_like(out_D2)
        loss_adv1 = self.loss_fn2(out_D1, source_label1)
        loss_adv2 = self.loss_fn2(out_D2, source_label2)

        loss = loss_seg2 + self.lambda_[0] * loss_seg1 + self.lambda_[2] * loss_adv2 + self.lambda_[1] * loss_adv1

        loss_seg1 = ops.stop_gradient(loss_seg1)
        loss_seg2 = ops.stop_gradient(loss_seg2)
        loss_adv1 = ops.stop_gradient(loss_adv1)
        loss_adv2 = ops.stop_gradient(loss_adv2)

        return loss, (loss_seg1, loss_seg2, loss_adv1, loss_adv2)


class WithLossCellD1(nn.Cell):
    def __init__(self, net_G, net_D1, loss_fn, size_source, size_target):
        super(WithLossCellD1, self).__init__(auto_prefix=True)
        self.net_G = net_G
        self.net_D1 = net_D1
        self.net_G.set_grad(False)
        self.net_D1.set_grad(True)
        self.size_source = size_source
        self.size_target = size_target
        self.interp_source = ops.ResizeBilinear(size=(size_source[1], size_source[0]))
        self.interp_target = ops.ResizeBilinear(size=(size_target[1], size_target[0]))
        self.loss_fn = loss_fn
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()
        self.softmax = Softmax(axis=1)

    def construct(self, image_source, label, image_target):
        pred1, _ = self.net_G(image_source)
        pred1 = self.interp_source(pred1)
        pred1 = self.softmax(pred1)
        pred1 = ops.stop_gradient(pred1)

        pred1_target, _ = self.net_G(image_target)
        pred1_target = self.interp_target(pred1_target)
        pred1_target = self.softmax(pred1_target)
        pred1_target = ops.stop_gradient(pred1_target)

        out_s, out_t = self.net_D1(pred1), self.net_D1(pred1_target)
        label_s, label_t = self.zeros_like(out_s), self.ones_like(out_t)

        loss1 = self.loss_fn(out_s, label_s)
        loss2 = self.loss_fn(out_t, label_t)

        return (loss1 + loss2) / 2.0


class WithLossCellD2(nn.Cell):
    def __init__(self, net_G, net_D2, loss_fn, size_source, size_target):
        super(WithLossCellD2, self).__init__(auto_prefix=True)
        self.net_G = net_G
        self.net_D2 = net_D2
        self.net_G.set_grad(False)
        self.net_D2.set_grad(True)
        self.size_source = size_source
        self.size_target = size_target
        self.interp_source = ops.ResizeBilinear(size=(size_source[1], size_source[0]))
        self.interp_target = ops.ResizeBilinear(size=(size_target[1], size_target[0]))
        self.loss_fn = loss_fn
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()
        self.softmax = Softmax(axis=1)

    def construct(self, image_source, label, image_target):
        _, pred2 = self.net_G(image_source)
        pred2 = self.interp_source(pred2)
        pred2 = self.softmax(pred2)
        pred2 = ops.stop_gradient(pred2)

        _, pred2_target = self.net_G(image_target)
        pred2_target = self.interp_target(pred2_target)
        pred2_target = self.softmax(pred2_target)
        pred2_target = ops.stop_gradient(pred2_target)

        out_s, out_t = self.net_D2(pred2), self.net_D2(pred2_target)
        label_s, label_t = self.zeros_like(out_s), self.ones_like(out_t)

        loss1 = self.loss_fn(out_s, label_s)
        loss2 = self.loss_fn(out_t, label_t)

        return (loss1 + loss2) / 2.0


class TrainOneStepCellG(nn.Cell):
    def __init__(self, network, optimizer):
        super(TrainOneStepCellG, self).__init__(auto_prefix=False)
        self.network = network
        # self.network.set_grad()
        self.optimizer = optimizer
        self.weight = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        out = self.network(*inputs)
        grads = self.grad(self.network, self.weight)(*inputs)
        out = ops.functional.depend(out, self.optimizer(grads))
        return out


class TrainOneStepCellD(nn.Cell):
    def __init__(self, network, optimizer):
        super(TrainOneStepCellD, self).__init__(auto_prefix=False)
        self.network = network  # 定义前向网络
        # self.network.set_grad()  # 构建反向网络图
        self.optimizer = optimizer  # 定义优化器
        self.weight = self.optimizer.parameters  # 获取更新的权重
        self.grad = ops.GradOperation(get_by_list=True)  # 定义梯度计算方法

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weight)(*inputs)
        loss = ops.functional.depend(loss, self.optimizer(grads))
        return loss


class AdaptSegNet(nn.Cell):
    def __init__(self, model_G, model_D1, model_D2, size_source, size_target, ):
        super(AdaptSegNet, self).__init__()
        self.model_G = model_G
        self.model_D1 = model_D1
        self.model_D2 = model_D2
        self.size_source = size_source
        self.size_target = size_target
        self.interp_source = ops.ResizeBilinear(size=(size_source[1], size_source[0]))
        self.interp_target = ops.ResizeBilinear(size=(size_target[1], size_target[0]))
        self.softmax = Softmax(axis=1)

    def construct(self, image_source, image_target):
        # generator
        # image_source, image_target = data[:, :3, :, :], data[:, 3:, :, :]
        # image_source, image_target = data[:, 0, :, :, :], data[:, 1, :, :, :]
        pred1, pred2 = self.model_G(image_source)
        pred1_target, pred2_target = self.model_G(image_target)
        pred1 = self.interp_source(pred1)
        pred2 = self.interp_source(pred2)
        pred1_target = self.interp_target(pred1_target)
        pred2_target = self.interp_target(pred2_target)

        D_out1_part1 = self.model_D1(self.softmax(pred1_target))
        D_out2_part1 = self.model_D2(self.softmax(pred2_target))

        # Adversarial
        pred1_part2 = ops.stop_gradient(self.softmax(pred1))
        pred2_part2 = ops.stop_gradient(self.softmax(pred2))
        pred1_target_part2 = ops.stop_gradient(self.softmax(pred1_target))
        pred2_target_part2 = ops.stop_gradient(self.softmax(pred2_target))
        out_s1_part2, out_t1_part2 = self.model_D1(pred1_part2), self.model_D1(pred1_target_part2)
        out_s2_part2, out_t2_part2 = self.model_D2(pred2_part2), self.model_D2(pred2_target_part2)

        return (pred1, pred2, D_out1_part1, D_out2_part1), (out_s1_part2, out_s2_part2, out_t1_part2, out_t2_part2)


def get_adaptsegnet(config):
    model_G = get_deeplab_v2(num_classes=config.num_classes)
    model_D1 = FCDiscriminator(num_classes=config.num_classes)
    model_D2 = FCDiscriminator(num_classes=config.num_classes)
    return AdaptSegNet(model_G, model_D1, model_D2, config.input_size, config.input_size_target)


class NetWithLoss(nn.WithLossCell):
    def __init__(self, net: nn.Cell, loss_fn: nn.Cell):
        super(NetWithLoss, self).__init__(net, loss_fn)

    def construct(self, s_image, t_image, s_label):
        out = self._backbone(s_image, t_image)
        return self._loss_fn(out, s_label)


class CustomTrainOneStepCell(nn.Cell):
    def __init__(self, net, loss_fn,
                 optimizer_G: nn.Optimizer, optimizer_D1: nn.Optimizer, optimizer_D2: nn.Optimizer,
                 sens=1.0):
        super(CustomTrainOneStepCell, self).__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.net_with_loss = NetWithLoss(self.net, self.loss_fn)
        self.optimizer_G = optimizer_G
        self.optimizer_D1 = optimizer_D1
        self.optimizer_D2 = optimizer_D2

        self.weights = ParameterTuple((*self.optimizer_G.parameters, *self.optimizer_D1.parameters, *self.optimizer_D2.parameters))
        self.len_grad_G = len(self.optimizer_G.parameters)
        self.len_grad_D1 = len(self.optimizer_D1.parameters)
        self.len_grad_D2 = len(self.optimizer_D2.parameters)
        self.weight_G = ParameterTuple(self.optimizer_G.parameters)
        self.weight_D1 = ParameterTuple(self.optimizer_D1.parameters)
        self.weight_D2 = ParameterTuple(self.optimizer_D2.parameters)
        self.grad = ops.GradOperation(get_by_list=True)
        self.sens = sens
        self.print = ops.Print()
        self.get_grad_reducer()
        self.loss_part1, self.loss_part2, self.loss_part3 = mindspore.Tensor(1.), mindspore.Tensor(2.), mindspore.Tensor(3.)

    def get_grad_reducer(self):
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            if isinstance(self.optimizer, (nn.AdaSumByGradWrapCell, nn.AdaSumByDeltaWeightWrapCell)):
                from mindspore.communication.management import get_group_size, create_group, get_rank
                group_number = get_group_size() // 8
                self.degree = int(self.degree / group_number)
                group_list = [list(range(x * self.degree, (x + 1) * self.degree)) for x in range(group_number)]
                current_index = get_rank() // 8
                server_group_name = "allreduce_" + str(current_index)
                create_group(server_group_name, group_list[current_index])
                self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree,
                                                           group=server_group_name)
            else:
                self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree)

    def grad_cal(self, net_with_loss, weights, *inputs):
        grads = self.grad(net_with_loss, weights)(*inputs)
        grads = self.grad_reducer(grads)
        return grads

    def construct(self, *inputs, **kwargs):
        # loss = self.net_with_loss(*inputs)
        # loss_part1, loss_part2, loss_part3 = loss
        loss_part1, loss_part2, loss_part3 = self.loss_part1, self.loss_part2, self.loss_part3
        # print('self.weight_G :', len(self.weight_G))
        # print('self.weight_D1:', len(self.weight_D1))
        # print('self.weight_D2:', len(self.weight_D2))

        # grads_G = self.grad_cal(self.net_with_loss, self.weight, *inputs)
        # grads_D1 = self.grad_cal(self.net_with_loss, self.weight_D1, *inputs)
        # grads_D2 = self.grad_cal(self.net_with_loss, self.weight_D2, *inputs)

        # self.grad = ops.GradOperation(get_by_list=True)
        # grads_G = self.grad(self.net_with_loss, self.weight_G)(*inputs)
        # self.grad = ops.GradOperation(get_by_list=True)
        # grads_D1 = self.grad(self.net_with_loss, self.weight_D1)(*inputs)
        # self.grad = ops.GradOperation(get_by_list=True)
        # grads_D2 = self.grad(self.net_with_loss, self.weight_D2)(*inputs)

        # grads = self.grad_cal(self.net_with_loss, self.weights, *inputs)
        # grads_G = grads[:self.len_grad_G]
        # grads_D1 = grads[self.len_grad_G:self.len_grad_G + self.len_grad_D1]
        # grads_D2 = grads[-self.len_grad_D2:]

        # print('grad_G :', len(grads_G))
        # print('grad_D1:', len(grads_D1))
        # print('grad_D2:', len(grads_D2))

        # loss_part1 = F.depend(loss_part1, self.optimizer_G(grads_G))
        # loss_part2 = F.depend(loss_part2, self.optimizer_D1(grads_D1))
        # loss_part3 = F.depend(loss_part3, self.optimizer_D2(grads_D2))
        # print(loss_part1, loss_part2, loss_part3)
        return loss_part1, loss_part2, loss_part3





def get_TrainOneStepCell(config, net, loss_fn):
    learning_rate = nn.PolynomialDecayLR(learning_rate=config.learning_rate, end_learning_rate=1e-9, decay_steps=config.num_steps, power=config.power)
    optimizer = nn.SGD(net.model_G.trainable_params(), learning_rate=learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

    learning_rate_D1 = nn.PolynomialDecayLR(learning_rate=config.learning_rate_D, end_learning_rate=1e-9, decay_steps=config.num_steps, power=config.power)
    optimizer_D1 = nn.Adam(net.model_D1.trainable_params(), learning_rate=learning_rate_D1, beta1=0.9, beta2=0.99)

    learning_rate_D2 = nn.PolynomialDecayLR(learning_rate=config.learning_rate_D, end_learning_rate=1e-9, decay_steps=config.num_steps, power=config.power)
    optimizer_D2 = nn.Adam(net.model_D2.trainable_params(), learning_rate=learning_rate_D2, beta1=0.9, beta2=0.99)

    return CustomTrainOneStepCell(net, loss_fn, optimizer, optimizer_D1, optimizer_D2)
