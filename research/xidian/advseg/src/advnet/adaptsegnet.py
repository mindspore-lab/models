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

from mindspore import ops, ParameterTuple, nn
from mindspore.ops import functional as F
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode, _get_enable_parallel_optimizer)
from mindspore.nn.wrap import DistributedGradReducer
from mindspore.context import ParallelMode

from src.model_utils import Softmax, split_checkpoint
from .deeplab import *
from .deeplab_multi import *
from .deeplabv2 import *
from .discriminator import *


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
        self.softmax = nn.Softmax(axis=1)

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

        return pred1, pred2, D_out1_part1, D_out2_part1, pred1_target, pred2_target


class WithLossG(nn.Cell):
    def __init__(self, net: AdaptSegNet, loss_fn1, loss_fn2, lambda_):
        super().__init__()
        self.net = net
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.lambda_ = lambda_
        self.zeros_like = ops.ZerosLike()
        self.stop_grad = ops.stop_gradient

    def construct(self, s_image, t_image, s_label):
        pred1, pred2, D_out1_part1, D_out2_part1, pred1_target, pred2_target = self.net(s_image, t_image)
        loss_seg1 = self.loss_fn1(pred1, s_label)
        loss_seg2 = self.loss_fn1(pred2, s_label)
        loss_adv1 = self.loss_fn2(D_out1_part1, self.zeros_like(D_out1_part1))
        loss_adv2 = self.loss_fn2(D_out2_part1, self.zeros_like(D_out2_part1))
        loss_G = loss_seg2 + self.lambda_[0] * loss_seg1 + self.lambda_[2] * loss_adv2 + self.lambda_[1] * loss_adv1

        pred1 = self.stop_grad(pred1)
        pred2 = self.stop_grad(pred2)
        pred1_target = self.stop_grad(pred1_target)
        pred2_target = self.stop_grad(pred2_target)
        return loss_G, (pred1, pred2, pred1_target, pred2_target)


class TrainOneStepG(nn.Cell):
    def __init__(self, net_with_loss: WithLossG, optimizer, sens=1.0):
        super().__init__()
        self.net_with_loss = net_with_loss
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)
        self.sens = sens
        self.print = ops.Print()
        self.get_grad_reducer()

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

    # output=(pred1, pred2, pred1_target, pred2_target)
    def construct(self, *inputs):
        loss_G, output = self.net_with_loss(*inputs)
        grads = self.grad(self.net_with_loss, self.weights)(*inputs)
        grads = self.grad_reducer(grads)
        loss_G = F.depend(loss_G, self.optimizer(grads))
        return loss_G, output


class WithlossD(nn.Cell):
    def __init__(self, model_D, loss_fn):
        super().__init__()
        self.model_D = model_D
        self.loss_fn = loss_fn
        self.softmax = nn.Softmax(axis=1)
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()

    def construct(self, pred, pred_target):
        prob = self.softmax(pred)
        prob_target = self.softmax(pred_target)
        out = self.model_D(prob)
        out_target = self.model_D(prob_target)
        loss = (self.loss_fn(out, self.zeros_like(out)) + self.loss_fn(out_target, self.ones_like(out_target))) / 2
        return loss


class TrainOneStepD(nn.Cell):
    def __init__(self, net_with_loss, optimizer, sens=1.0):
        super().__init__()
        self.net_with_loss = net_with_loss
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)
        self.sens = sens
        self.print = ops.Print()
        self.get_grad_reducer()

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

    def construct(self, *inputs):
        loss = self.net_with_loss(*inputs)
        grads = self.grad(self.net_with_loss, self.weights)(*inputs)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss


class CustomTrainOneStep(nn.Cell):
    def __init__(self, Train_G: TrainOneStepG, Train_D1: TrainOneStepD, Train_D2: TrainOneStepD):
        super().__init__()
        self.Train_G = Train_G
        self.Train_D1 = Train_D1
        self.Train_D2 = Train_D2

    def construct(self, *inputs):
        loss_G, (pred1, pred2, pred1_target, pred2_target) = self.Train_G(*inputs)
        loss_D1 = self.Train_D1(pred1, pred1_target)
        loss_D2 = self.Train_D2(pred2, pred2_target)
        return loss_G, loss_D1, loss_D2


def get_adaptsegnetCell(config):
    model_G = get_deeplab_v2(num_classes=config.num_classes)
    model_D1 = FCDiscriminator(num_classes=config.num_classes)
    model_D2 = FCDiscriminator(num_classes=config.num_classes)

    if config.restore_from:
        print('load model from : {}'.format(config.restore_from))
        saved_state_dict = mindspore.load_checkpoint(config.restore_from)
        split_list = ['net_G', 'net_D1', 'net_D2']
        train_state_dict = split_checkpoint(saved_state_dict, split_list=split_list)
        mindspore.load_param_into_net(model_G, train_state_dict['net_G'], strict_load=True)
        print('success load model !')

    return AdaptSegNet(model_G, model_D1, model_D2, config.input_size, config.input_size_target)


def get_TrainOneStepCell(config, net: AdaptSegNet, loss_fn1, loss_fn2):
    net_with_lossG = WithLossG(net, loss_fn1, loss_fn2, config.lambda_)
    net_with_lossD1 = WithlossD(net.model_D1, loss_fn2)
    net_with_lossD2 = WithlossD(net.model_D2, loss_fn2)

    learning_rate = nn.PolynomialDecayLR(learning_rate=config.learning_rate, end_learning_rate=1e-9, decay_steps=config.num_steps, power=config.power)
    optimizer = nn.SGD(net.model_G.trainable_params(), learning_rate=learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

    learning_rate_D1 = nn.PolynomialDecayLR(learning_rate=config.learning_rate_D, end_learning_rate=1e-9, decay_steps=config.num_steps, power=config.power)
    optimizer_D1 = nn.Adam(net.model_D1.trainable_params(), learning_rate=learning_rate_D1, beta1=0.9, beta2=0.99)

    learning_rate_D2 = nn.PolynomialDecayLR(learning_rate=config.learning_rate_D, end_learning_rate=1e-9, decay_steps=config.num_steps, power=config.power)
    optimizer_D2 = nn.Adam(net.model_D2.trainable_params(), learning_rate=learning_rate_D2, beta1=0.9, beta2=0.99)

    net_trainG = TrainOneStepG(net_with_lossG, optimizer)
    net_trainD1 = TrainOneStepD(net_with_lossD1, optimizer_D1)
    net_trainD2 = TrainOneStepD(net_with_lossD2, optimizer_D2)
    return CustomTrainOneStep(net_trainG, net_trainD1, net_trainD2)
