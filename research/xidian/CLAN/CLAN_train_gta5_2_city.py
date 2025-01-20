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
import numpy as np
import time
import os
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor, Parameter, context
import mindspore.common.initializer as init
from options.train_options import TrainOptions
from model.CLAN_G import Res_Deeplab
from model.CLAN_D import FCDiscriminator
from utils import WeightedBCEWithLogitsLoss, SoftmaxCrossEntropyLoss, Softmax
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from CLAN_evaluate_city import evaluation
from utils.loss import WeightedBCEWithLogitsLoss
from mindspore import dtype as mstype


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # early stopping
PREHEAT_STEPS = int(NUM_STEPS_STOP / 20)
Epsilon = 0.4


def split_checkpoint(checkpoint, split_list=None):
    if split_list == None:
        return checkpoint
    checkpoint_dict = {name: {} for name in split_list}
    for key, value in checkpoint.items():
        prefix = key.split('.')[0]
        if prefix not in checkpoint_dict:
            checkpoint_dict[key] = value.asnumpy()
            continue
        name = key.replace(prefix + '.', '')
        checkpoint_dict[prefix][name] = value
    return checkpoint_dict

class WithLossCellG(nn.Cell):
    def __init__(self, lambda_, net_G, net_D, loss_seg, loss_weight_bce, loss_bce, size_source, size_target,
                 batch_size=1,
                 num_classes=19):
        super(WithLossCellG, self).__init__(auto_prefix=True)
        self.lambda_ = lambda_
        self.net_G = net_G
        self.net_D = net_D
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.size_source = size_source
        self.size_target = size_target
        self.interp_source = ops.ResizeBilinear(size=(size_source[1], size_source[0]))
        self.interp_target = ops.ResizeBilinear(size=(size_target[1], size_target[0]))
        self.loss_seg = loss_seg
        self.loss_weight_bce = loss_weight_bce
        self.loss_bce = loss_bce
        self.softmax = Softmax(axis=1)
        self.zeros_like = ops.ZerosLike()
        self.reshape = ops.Reshape()
        self.sum = ops.ReduceSum()
        self.norm_1 = ops.LpNorm(axis=1, p=2)
        self.norm_0 = ops.LpNorm(axis=0, p=2)
        self.mul = ops.Mul()
        self.cast = ops.Cast()

    # @staticmethod
    def module_param_flatten(self, module):

        return ops.Concat(axis=0)([self.reshape(param, (-1,)) for param in module.get_parameters()])

    # @staticmethod
    def weightmap(self, pred1, pred2, out_D):
        output = 1.0 - self.reshape(self.sum((pred1 * pred2), 1), (1, 1, pred1.shape[2], pred1.shape[3])) \
                 / self.reshape((self.norm_1(pred1) * self.norm_1(pred2)), (1, 1, pred1.shape[2], pred1.shape[3]))

        output = ops.ResizeBilinear(size=(out_D.shape[2], out_D.shape[3]))(output)

        return output

    def construct(self, image_source, label, image_target, i_iter):

        self.net_G.requires_grad = True
        self.net_D.requires_grad = False
        pred1, pred2 = self.net_G(image_source)
        pred1 = self.interp_source(pred1)
        pred2 = self.interp_source(pred2)
        loss_seg_1 = self.loss_seg(pred1, label)
        loss_seg_2 = self.loss_seg(pred2, label)

        loss_seg = loss_seg_1 + loss_seg_2

        pred1_target, pred2_target = self.net_G(image_target)
        pred1_target = self.interp_target(pred1_target)
        pred2_target = self.interp_target(pred2_target)

        pred_target = self.softmax(pred1_target + pred2_target)

        W5 = self.module_param_flatten(self.net_G.layer5)
        W6 = self.module_param_flatten(self.net_G.layer6)

        loss_weight = 1 - ops.cosine_similarity(W5, W6, dim=0)
        # loss_weight = (self.sum(self.mul(W5, W6))  # todo
        #                / (self.norm_0(W5) * self.norm_0(W6)) + 1)  # +1 is for a positive loss

        pred1_target_sm = self.softmax(pred1_target)
        pred2_target_sm = self.softmax(pred2_target)

        out_D = self.net_D(pred_target)
        source_label = self.zeros_like(out_D)

        weight_map = self.weightmap(pred1_target_sm, pred2_target_sm, out_D)


        if (i_iter > 2000):
            loss_adv_G = self.loss_weight_bce(out_D, source_label, weight_map, Epsilon, self.lambda_[2])
        else:
            loss_adv_G = self.loss_bce(out_D, source_label)

        loss = loss_seg + self.lambda_[1] * loss_adv_G + loss_weight * self.lambda_[0]


        loss_seg = ops.stop_gradient(loss_seg)
        loss_adv_G = ops.stop_gradient(loss_adv_G)
        loss_weight = ops.stop_gradient(loss_weight)

        return loss, (loss_seg, loss_adv_G, loss_weight)


class WithLossCellD(nn.Cell):
    def __init__(self, lambda_, net_G, net_D, loss_bce, loss_weight_bce, size_source, size_target):
        super(WithLossCellD, self).__init__(auto_prefix=True)
        self.lambda_ = lambda_
        self.net_G = net_G
        self.net_D = net_D
        self.size_source = size_source
        self.size_target = size_target
        self.interp_source = ops.ResizeBilinear(size=(size_source[1], size_source[0]))
        self.interp_target = ops.ResizeBilinear(size=(size_target[1], size_target[0]))
        self.loss_bce = loss_bce
        self.loss_weight_bce = loss_weight_bce
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()
        self.softmax = Softmax(axis=1)
        self.reshape = ops.Reshape()
        self.sum = ops.ReduceSum()
        self.norm_1 = ops.LpNorm(axis=1, p=2)
        self.norm_0 = ops.LpNorm(axis=0, p=2)
        self.mul = ops.Mul()

    def weightmap(self, pred1, pred2, out_t):
        output = 1.0 - self.reshape(self.sum((pred1 * pred2), 1), (1, 1, pred1.shape[2], pred1.shape[3])) \
                 / self.reshape((self.norm_1(pred1) * self.norm_1(pred2)), (1, 1, pred1.shape[2], pred1.shape[3]))

        output = ops.ResizeBilinear(size=(out_t.shape[2], out_t.shape[3]))(output)

        return output

    def construct(self, image_source, image_target, i_iter):

        self.net_G.requires_grad = False
        self.net_D.requires_grad = True

        pred1, pred2 = self.net_G(image_source)
        pred1 = self.interp_source(pred1)
        pred2 = self.interp_source(pred2)
        pred = self.softmax(pred1 + pred2)
        pred = ops.stop_gradient(pred)

        pred1_target, pred2_target = self.net_G(image_target)
        pred1_target = self.interp_target(pred1_target)
        pred2_target = self.interp_target(pred2_target)
        pred_target = self.softmax(pred1_target + pred2_target)
        pred_target = ops.stop_gradient(pred_target)

        pred1_target_sm = self.softmax(pred1_target)
        pred2_target_sm = self.softmax(pred2_target)

        out_s, out_t = self.net_D(pred), self.net_D(pred_target)
        label_s, label_t = self.zeros_like(out_s), self.ones_like(out_t)

        loss_D_s = self.loss_bce(out_s, label_s)

        weight_map = self.weightmap(pred1_target_sm, pred2_target_sm, out_t)


        if (i_iter > 2000):
            loss_adv_D = self.loss_weight_bce(out_t, label_t, weight_map, Epsilon, self.lambda_[2])
        else:
            loss_adv_D = self.loss_bce(out_t, label_t)

        loss_D = loss_D_s + self.lambda_[1] * loss_adv_D

        loss_D_s = ops.stop_gradient(loss_D_s)
        loss_adv_D = ops.stop_gradient(loss_adv_D)

        return loss_D, (loss_D_s, loss_adv_D)


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
        self.optimizer = optimizer  # 定义优化器
        self.weight = self.optimizer.parameters  # 获取更新的权重
        self.grad = ops.GradOperation(get_by_list=True)  # 定义梯度计算方法

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weight)(*inputs)
        loss = ops.functional.depend(loss, self.optimizer(grads))
        return loss


def main():

    opt = TrainOptions()
    args = opt.initialize()

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)
    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)



    if args.device == 'ascend':
        mindspore.set_context(mode=mindspore.GRAPH_MODE, device_id=5, device_target="Ascend")

    elif args.device == 'gpu':
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU")
    else:
        context.set_context(mode=context.PYNATIVE_MODE, save_graphs=False, device_target="CPU")

    print('设备：', args.device)

    if args.debug:
        args.batch_size = 1
        input_size = (128, 128)
        input_size_target = (128, 128)
        args.save_pred_every = 10
        args.num_steps_stop = 100
        args.not_val = False


    # [Part 1: 加载模型]
    model = Res_Deeplab(num_classes=args.num_classes)

    #加载预训练模型
    if args.restore_from:

        param_dict = mindspore.load_checkpoint(args.restore_from)
        net_params = model.parameters_dict()

        filtered_param_dict = {}
        for param_name, param_value in param_dict.items():
            param_shape = param_value.shape
            matching_params = [p for p in net_params.values() if p.shape == param_shape]

            if matching_params:
                filtered_param_dict[param_name] = param_value
            else:
                print(f"Warning: Parameter shape mismatch. Skipping parameter with shape {param_shape}.")

        split_list = ['net_G', 'net_D']
        train_state_dict = split_checkpoint(filtered_param_dict, split_list=split_list)

        mindspore.load_param_into_net(model, train_state_dict['net_G'])
        print('success load model !')

    #初始化判别器
    model_D = FCDiscriminator(num_classes=args.num_classes)

    parameters = model.trainable_params()
    parameters_D = model_D.trainable_params()
    print('model_G:', len(parameters))
    print('model_D:', len(parameters_D))

    # [Part 2: 优化器和损失函数]
    learning_rate = nn.PolynomialDecayLR(learning_rate=args.learning_rate, end_learning_rate=1e-9,
                                         decay_steps=args.num_steps, power=args.power)
    optimizer = nn.SGD(model.trainable_params(),
                       learning_rate=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    learning_rate_D = nn.PolynomialDecayLR(learning_rate=args.learning_rate, end_learning_rate=1e-9,
                                           decay_steps=args.num_steps, power=args.power)
    optimizer_D = nn.Adam(model_D.trainable_params(), learning_rate=learning_rate_D, beta1=0.9, beta2=0.99)

    loss_seg = SoftmaxCrossEntropyLoss()
    loss_weight_bce = WeightedBCEWithLogitsLoss()
    loss_bce = nn.BCEWithLogitsLoss()

    iter_start = 0
    best_iou = 0.0
    if args.continue_train:
        print('continue training')
        filepath, filename = os.path.split(args.continue_train)
        print('filepath :', filepath)
        print('filename :', filename)
        target_path = filepath
        os.makedirs(target_path, exist_ok=True)
        logger = open(os.path.join(target_path, 'Train_log.log'), 'a')

        split_list = ['net_G', 'net_D']
        train_state_dict = mindspore.load_checkpoint(args.continue_train)
        train_state_dict = split_checkpoint(train_state_dict, split_list=split_list)
        iter_start = train_state_dict['iter']
        best_iou = train_state_dict['best_IoU']
        mindspore.load_param_into_net(model, train_state_dict['net_G'])
        mindspore.load_param_into_net(model_D, train_state_dict['net_D'])
        optimizer.global_step.set_data(mindspore.Tensor([iter_start], dtype=mindspore.int32))
        optimizer_D.global_step.set_data(mindspore.Tensor([iter_start], dtype=mindspore.int32))
    else:
        time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        target_path = os.path.join(args.snapshot_dir, time_now)
        os.makedirs(target_path, exist_ok=True)
        logger = open(os.path.join(target_path, 'Train_log.log'), 'w')

    # [Part 3: 加载数据]
    gta_genarator = GTA5DataSet(args.data_dir, args.data_list,
                                max_iters=args.num_steps * args.iter_size * args.batch_size,
                                crop_size=input_size,
                                mean=IMG_MEAN)
    gta_dataset = ds.GeneratorDataset(gta_genarator, shuffle=True, column_names=['image', 'label', 'size'])
    gta_dataset = gta_dataset.batch(batch_size=args.batch_size)
    train_iterator = gta_dataset.create_dict_iterator()

    cityscapes_generator = cityscapesDataSet(args.data_dir_target, os.path.join(args.devkit_dir, f'{args.set}.txt'),
                                             max_iters=args.num_steps * args.iter_size * args.batch_size,
                                             crop_size=input_size_target, scale=False,
                                             mean=IMG_MEAN,
                                             set=args.set)
    cityscapes_dataset = ds.GeneratorDataset(cityscapes_generator, shuffle=True,
                                             column_names=['image', 'size'])
    cityscapes_dataset = cityscapes_dataset.batch(batch_size=args.batch_size)
    target_iterator = cityscapes_dataset.create_dict_iterator()

    evaluation_generator = cityscapesDataSet(args.data_dir_target, os.path.join(args.devkit_dir, 'val.txt'),
                                             crop_size=input_size_target, scale=False,
                                             mean=IMG_MEAN,
                                             set='val')
    evaluation_dataset = ds.GeneratorDataset(evaluation_generator, shuffle=False,
                                             column_names=['image', 'size'])
    evaluation_dataset = evaluation_dataset.batch(batch_size=1)
    evaluation_iterator = evaluation_dataset.create_dict_iterator()


    lambda_ = [args.lambda_weight, args.lambda_adv, args.lambda_local]

    model_G_with_loss = WithLossCellG(lambda_=lambda_,
                                      net_G=model,
                                      net_D=model_D,
                                      loss_seg=loss_seg,
                                      loss_weight_bce=loss_weight_bce,
                                      loss_bce=loss_bce,
                                      size_source=input_size,
                                      size_target=input_size_target,
                                      batch_size=args.batch_size,
                                      num_classes=args.num_classes).set_grad()
    model_D_with_loss = WithLossCellD(lambda_=lambda_,
                                      net_G=model,
                                      net_D=model_D,
                                      loss_bce=loss_bce,
                                      loss_weight_bce=loss_weight_bce,
                                      size_source=input_size,
                                      size_target=input_size_target).set_grad()




    model_G_train = TrainOneStepCellG(model_G_with_loss, optimizer)
    model_D_train = TrainOneStepCellD(model_D_with_loss, optimizer_D)

    model_G_train.set_train()
    model_D_train.set_train()

    # start train
    time_start_all = time.time()
    time_start_one = time.time()
    time_start_log = time.time()

    print(f'训练启动：',
          '\n开始的代数：', iter_start,
          '\n最高IoU：', best_iou,
          '保存地址：', target_path
          )

    for i_iter in range(iter_start, args.num_steps):

        damping = (1 - i_iter / NUM_STEPS)

        s_data = next(train_iterator)
        image_s, label_s = s_data['image'], s_data['label']
        t_data = next(target_iterator)
        image_t = t_data['image']

        image_s, label_s = Tensor(image_s), Tensor(label_s)
        image_t = Tensor(image_t)

        _, (loss_seg, loss_adv_G, loss_weight) = model_G_train(image_s, label_s, image_t, i_iter, damping)

        (loss_seg, loss_adv_G, loss_weight) = \
            map(lambda x: x.asnumpy(), (loss_seg, loss_adv_G, loss_weight))


        _, (loss_D_s, loss_adv_D) = model_D_train(image_s, image_t, i_iter)

        (loss_D_s, loss_adv_D) = \
            map(lambda x: x.asnumpy(), (loss_D_s, loss_adv_D))

        time_end_one = time.time()
        print('iter = {0:8d}/{1:8d} loss_seg = {2:.6f} loss_adv_G = {3:.6f} loss_weight = {4:.6f} loss_D_s = {5:.6f} '
              'loss_adv_D = {6:.6f} time = {7:.6f}'.format(
            i_iter, args.num_steps, loss_seg, loss_adv_G, loss_weight, loss_D_s, loss_adv_D,
            time_end_one - time_start_one))
        time_start_one = time_end_one
        if i_iter % 1 == 0:
            time_end_log = time.time()
            logger.write(
                'iter = {0:8d}/{1:8d} loss_seg = {2:.6f} loss_adv_G = {3:.6f} loss_weight = {4:.6f} loss_D_s = {5:.6f} '
                'loss_adv_D = {6:.6f} time = {7:.6f}\n'.format(
                    i_iter, args.num_steps, loss_seg, loss_adv_G, loss_weight, loss_D_s, loss_adv_D,
                    time_end_log - time_start_log))
            time_start_log = time_end_log

        if (i_iter + 1) % args.save_pred_every == 0:
            print('val checkpoint ...')

            if args.not_val:
                miou = evaluation(model, evaluation_iterator, ops.ResizeBilinear(size=(1024, 2048)),
                                  args.data_dir_target,
                                  args.save_result_path, args.devkit_dir, logger=logger, save=False)
                miou = float(miou)
            else:
                miou = -0.1

            checkpoint_path = os.path.join(target_path, 'GTA5_' + str(i_iter + 1) + '.ckpt')

            param_list = [{'name': name, 'data': param} for name, param in
                          model.parameters_and_names(name_prefix='net_G')]
            for name, param in model_D.parameters_and_names(name_prefix='net_D'):
                param_list.append({'name': name, 'data': param})

            append_dict = {'iter': int(optimizer.global_step.asnumpy()[0]),
                           'mIoU': float(miou),
                           'best_IoU': float(best_iou) if miou < best_iou else float(miou)}
            mindspore.save_checkpoint(param_list, checkpoint_path, append_dict=append_dict)

            if miou > best_iou:
                best_iou = miou
                checkpoint_path = os.path.join(target_path, 'GTA5_best.ckpt')
                mindspore.save_checkpoint(param_list, checkpoint_path, append_dict=append_dict)
            print('Best mIoU:', best_iou)
            logger.write(f'Best mIoU:{best_iou}\n')
            if i_iter >= args.num_steps_stop - 1:
                checkpoint_path = os.path.join(target_path, 'GTA5_Over.ckpt')
                mindspore.save_checkpoint(param_list, checkpoint_path, append_dict=append_dict)
                break
    print('Train Over ! Save Over model ')

    time_end_all = time.time()
    logger.write(f'训练总用时：{time_end_all - time_start_all}')
    logger.close()


if __name__ == '__main__':
    main()

