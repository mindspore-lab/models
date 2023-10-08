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

import time
import random
import os
import sys
import numpy as np
import mindspore.dataset.vision as c_vision
import mindspore.dataset as ds
from mindspore import context, Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

from model_utils.config import config
from src.models import Class_classifier, Domain_classifier, Backbone
from src.data_loader import GetLoader
from src.train_cell import withlosscell_class,withlosscell_domain,withlosscell_d,TrainOneStep_cls,TrainOneStepDomain,TrainOneStepD
from eval import test

def run_train():
    start_time = time.time()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(device_id=device_id)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_dataset_name = config.source_dataset_name
    target_dataset_name =  config.target_dataset_name
    source_image_root =   os.path.join(current_dir, config.source_image_root)
    target_image_root =   os.path.join(current_dir, config.target_image_root)
    model_root = os.path.join(current_dir,config.model_root)
    lr = config.lr
    lr_backbone_s = config.lr_backbone_s
    weight_decay = config.weight_decay

    batch_size = config.batch_size
    n_pretrain_epochs = config.n_pretrain_epochs
    n_epoch = config.n_epoch

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)

    #source data preporcessing
    img_transform_source_list = [
        c_vision.RandomResize(28),
        c_vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        c_vision.HWC2CHW()
    ]

    mnist_dataset_dir = os.path.join(current_dir,"./dataset/MNIST")
    dataset_source = ds.MnistDataset(dataset_dir=mnist_dataset_dir, usage='train')
    dataset_source = dataset_source.map(operations=img_transform_source_list)
    dataloader_source = dataset_source.batch(batch_size=batch_size, drop_remainder=True)

    #target data preporcessing
    img_transform_target_list = [
        c_vision.RandomResize(28),
        c_vision.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        c_vision.HWC2CHW()
    ]
    train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

    dataset_target = GetLoader(
        data_root=os.path.join(target_image_root, 'mnist_m_train'),
        data_list=train_list
        # transform=img_transform_target
    )
    dataset_target = ds.GeneratorDataset(source=dataset_target, column_names=["image", "label"])
    dataset_target = dataset_target.map(operations=img_transform_target_list, input_columns=["image"],
                                        output_columns=["image"])
    dataloader_target = dataset_target.batch(batch_size, drop_remainder=True)

    backbone_s = Backbone()
    backbone_t = Backbone()
    class_classifier = Class_classifier()
    domain_classifier = Domain_classifier()

    optimizer_backbone_s = nn.Adam(backbone_s.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
    optimizer_backbone_t = nn.Adam(backbone_t.trainable_params(), learning_rate=lr_backbone_s, weight_decay=weight_decay)
    optimizer_class_classifier = nn.Adam(class_classifier.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
    optimizer_domain_classifier = nn.Adam(domain_classifier.trainable_params(), learning_rate=lr, weight_decay=weight_decay)

    loss_class = ops.NLLLoss()
    loss_domain = ops.NLLLoss()

    best_accu_t = 0.0

    #pretrain
    trainer_cls = withlosscell_class(backbone_s, class_classifier, loss_class)
    trainer_cls_container = TrainOneStep_cls(trainer_cls, optimizer_backbone_s, optimizer_class_classifier)
    trainer_cls_container.set_train(True)

    best_acct = 0.0
    best_accs = 0.0
    setting_time = time.time()
    print("setting time: %.3f seconds"% (setting_time - start_time))

    pretrain_start_time = time.time()
    for epoch in range(n_pretrain_epochs):
        len_dataset_source = dataloader_source.get_dataset_size()
        dataset_source_iter = dataloader_source.create_dict_iterator()
        for i in range(len_dataset_source):
            data_source = next(dataset_source_iter)
            s_img, s_label = data_source['image'], data_source['label']
            s_img = ms.Tensor(s_img, ms.float32)
            s_label = Tensor(s_label, ms.int32)
            batch_size = len(s_label)
            weightClass = Tensor(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(np.float32))
            loss_s_class = trainer_cls_container(s_img, s_label, weightClass)
            sys.stdout.write('\r' + 'epoch :%d,iter: %d, loss_s_class: %.4f' % (epoch, i, loss_s_class))
            sys.stdout.flush()

    pretrain_end_time = time.time()
    print("\n")
    print("pretrain time: %.3f seconds"% (pretrain_end_time - pretrain_start_time))
    print("Average time per epoch in pretrain: %.3f seconds"% ((pretrain_end_time - pretrain_start_time)/n_pretrain_epochs))
    print("Average time per iteration in pretrain: %.3f seconds"% ((pretrain_end_time - pretrain_start_time)/(n_pretrain_epochs*len_dataset_source)))
    check_path_backbone = os.path.join(model_root, 'backbone_pretrain.ckpt')
    check_path_cls = os.path.join(model_root, 'class_classifier.ckpt')
    ms.save_checkpoint(backbone_s, check_path_backbone)
    ms.save_checkpoint(class_classifier, check_path_cls)
    acc = test(source_dataset_name, check_path_backbone)
    print('source domain pretrain accuracy: %f' % acc)
    print('source domain pretraining finished')

    # 读取预训练模型，将参数赋值给backbone_t
    param_dict_backbone = ms.load_checkpoint(check_path_backbone)
    ms.load_param_into_net(backbone_t, param_dict_backbone)

    trainer_domain = withlosscell_domain(backbone_s, backbone_t, domain_classifier, loss_domain)
    # 只更新domain_classifier的参数
    trainer_domain_container = TrainOneStepDomain(trainer_domain, optimizer_domain_classifier)
    trainer_domain_container.set_train(True)

    trainer_domain_t = withlosscell_d(backbone_t, domain_classifier, loss_domain)
    trainer_domain_t_container = TrainOneStepD(trainer_domain_t, optimizer_backbone_t, optimizer_domain_classifier)
    trainer_domain_t_container.set_train(True)

    main_train_start_time = time.time()
    for epoch in range(n_epoch):
        len_dataloader = min(dataloader_source.get_dataset_size(),
                             dataloader_target.get_dataset_size())  # min(len(dataloader_source), len(dataloader_target)) 461

        data_source_iter = dataloader_source.create_dict_iterator()  # iter(dataloader_source)
        data_target_iter = dataloader_target.create_dict_iterator()  # iter(dataloader_target)
        for i in range(len_dataloader):
            # training model using source data
            data_source = next(data_source_iter)
            s_img, s_label = data_source['image'], data_source['label']
            s_img = ms.Tensor(s_img, ms.float32)
            s_label = Tensor(s_label, ms.int32)

            batch_size = len(s_label)
            domain_label = ms.numpy.zeros(batch_size)  # 0表示源域
            domain_label = Tensor(domain_label, ms.int32)

            # training model using target data
            data_target = next(data_target_iter)
            t_img = data_target['image']
            t_img = ms.Tensor(t_img, ms.float32)

            batch_size = len(t_img)

            domain_label2 = ms.numpy.ones(batch_size)  # 1表示目标域
            domain_label2 = Tensor(domain_label2, ms.int32)
            # 以下是设置在分类时各个类别的权重系数，这里都取相同的权重
            weightClass = Tensor(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(np.float32))
            weightDomain = Tensor(np.array([1, 1]).astype(np.float32))

            loss_D_class = trainer_domain_container(s_img, domain_label, t_img, domain_label2, weightDomain)
            loss_G_class = trainer_domain_t_container(t_img, domain_label, weightDomain)
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d],  err_D_domain: %f, err_G_domain: %f,err_sum:%f' \
                             % (epoch, i + 1, len_dataloader,
                                loss_D_class.asnumpy(), loss_G_class.asnumpy(),
                                loss_D_class.asnumpy() + loss_G_class.asnumpy()))
            sys.stdout.flush()
            check_path_backbone_t = os.path.join(model_root, 'backbone_t.ckpt')
            check_path_cls = os.path.join(model_root, 'class_classifier.ckpt')
            check_path_domain_cls = os.path.join(model_root, 'domain_classifier.ckpt')
            ms.save_checkpoint(backbone_t, check_path_backbone_t)
            ms.save_checkpoint(class_classifier, check_path_cls)
            ms.save_checkpoint(domain_classifier, check_path_domain_cls)
        print('\n')
        accu_s = test(source_dataset_name, check_path_backbone)
        print('source domain accuracy: %f' % accu_s)
        accu_t = test(target_dataset_name, check_path_backbone_t)
        print('target domain accuracy: %f' % accu_t)
        if accu_t > best_acct:
            best_acct = accu_t
            best_accs = accu_s
            check_path_backbone = os.path.join(model_root, 'best_backbone_t.ckpt')
            check_path_cls = os.path.join(model_root, 'best_class_classifier.ckpt')
            check_path_domain_cls = os.path.join(model_root, 'best_domain_classifier.ckpt')
            ms.save_checkpoint(backbone_t, check_path_backbone)
            ms.save_checkpoint(class_classifier, check_path_cls)
            ms.save_checkpoint(domain_classifier, check_path_domain_cls)
    main_train_end_time = time.time()
    print("main train time : %.5f seconds" % (main_train_end_time - main_train_start_time))
    print("Average time per epoch in main train : %.5f seconds" % ((main_train_end_time - main_train_start_time) / n_epoch))
    print("Average time per iteration in main train : %.5f seconds" % ((main_train_end_time - main_train_start_time) / n_epoch / len_dataloader))
    print('============ Summary ============= \n')
    print('Accuracy of the %s dataset: %f' % ('mnist', best_accs))
    print('Accuracy of the %s dataset: %f' % ('mnist_m', best_acct))
    end_time = time.time()
    print('Total time: %f 秒' % (end_time - start_time))


if __name__ == '__main__':
    run_train()