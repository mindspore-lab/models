import argparse

import matplotlib.pyplot as plt
import mindspore
import numpy as np
import mindspore.dataset as ds
import mindspore as ms
import mindspore.ops as ops
from mindvision.engine.callback import ValAccMonitor
from mindvision.classification.models.head import DenseHead
from mindspore import nn
from data_list import ImageList, GetDatasetGenerator
import os
# import loss as loss_func
import network2
from mindspore import Tensor
from mindvision.engine.callback import LossMonitor
import mindspore.dataset.vision as vision
from mindspore import context, Tensor, ParameterTuple
from mindspore import nn, Model, ops
from mindspore import dtype as mstype

import warnings
warnings.filterwarnings("ignore")

# 动态图
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


def create_dataset_imagenet(dataset_path):
    """数据加载"""
    data_set = ds.ImageFolderDataset(dataset_path,
                                     num_parallel_workers=4,
                                     shuffle=False,
                                     decode=True)

    # 数据增强操作
    transform_img = [
        ds.vision.c_transforms.Resize((32,32)),
        ds.vision.c_transforms.Normalize((0.5,), (0.5,)),
        ds.vision.c_transforms.CenterCrop((32,32)),
        ds.vision.c_transforms.HWC2CHW(),
        # lambda x: ((x / 255).astype("float32"), np.random.normal(size=(100, 1, 1)).astype("float32"))
        ]

    # 数据映射操作
    data_set = data_set.map(input_columns="image",
                            num_parallel_workers=4,
                            operations=transform_img)

    # 批量操作
    data_set = data_set.batch(32, drop_remainder=True)
    return data_set


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CDAN SVHN MNIST')
    parser.add_argument('--method', type=str, default='CDAN', choices=['CDAN', 'CDAN-E', 'DANN'])
    parser.add_argument('--task', default='USPS2MNIST', help='task to perform')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='cuda device id')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--random', type=bool, default=False,
                        help='whether to use random')
    args = parser.parse_args()

    #mindspore.dataset.config.set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # -------------------------------------------------------------------------------------------------------------#
    # Load in train data
    # 获取处理后的数据集
    image_folder_dataset_dir = "../data/svhn2mnist/svhn_image"
    source_dataset = create_dataset_imagenet(image_folder_dataset_dir)
    image_folder_dataset_dir1 = "../data/svhn2mnist/mnist_train_image"
    source_dataset1 = create_dataset_imagenet(image_folder_dataset_dir1)
    image_folder_dataset_dir2 = "../data/svhn2mnist/mnist_test_image"
    test_dataset = create_dataset_imagenet(image_folder_dataset_dir2)

    # 获取数据集大小
    step_size = source_dataset.get_dataset_size()
    data_iter = next(source_dataset.create_dict_iterator(output_numpy=True))
    # # 可视化部分训练数据
    plt.figure(figsize=(10, 3), dpi=140)
    for i, image in enumerate(data_iter['image'][:30], 1):
        plt.subplot(3, 10, i)
        plt.axis("off")
        image = image / image.max()
        plt.imshow(image.transpose(1, 2, 0))
    plt.show()
    # image_folder_dataset_dir = "../data/svhn2mnist/svhn_image"
    # source_dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir, num_parallel_workers=4)
    #
    # image_folder_dataset_dir = "../data/svhn2mnist/mnist_train_image"
    # target_dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir, num_parallel_workers=4)
    #
    # image_folder_dataset_dir = "../data/svhn2mnist/mnist_test_image"
    # test_dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir, num_parallel_workers=4)

    # train_data = ImageList(open(source_list).readlines(), mode='RGB')
    # train_loader = ds.GeneratorDataset(train_data, ["data", "label"], shuffle=False)
    #
    # test_data = ImageList(open(test_list).readlines(), mode='RGB')
    # test_loader = ds.GeneratorDataset(train_data, ["data", "label"], shuffle=False)

    # img, label = source_dataset.__getitem__(1)

    # Test if the data is loaded
    # step_size = train_loader.source_len
    # train_loader = train_loader.batch(32)
    # batch_size = train_loader.get_batch_size()

    # step_size = train_loader.source_len
    '''''
    # -------------------------------------------------------------------------------------------------------------#
    # 构建训练网络
    # 定义优化器和损失函数
    network = network2.DTN()
    num_epochs = 40
    lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.001, total_step=step_size * num_epochs,
                            step_per_epoch=step_size, decay_epoch=num_epochs)
    opt = nn.SGD(params=network.trainable_params(), learning_rate=lr, weight_decay=0.0005, momentum=0.9)
    # opt = nn.Momentum(params=network.trainable_params(), learning_rate=lr, momentum=0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    # 手动构建训练网络
    loss_net = network2.CustomWithLossCell(network, loss)
    train_net = network2.CustomTrainOneStepCell(loss_net, opt)
    # 构建训练网络
    model = ms.Model(train_net)
    # 执行模型训练
    model.train(epoch=num_epochs, train_dataset=train_loader, callbacks=[LossMonitor(0.01)])
    # -------------------------------------------------------------------------------------------------------------#
    '''''
    #  网络训练
    network = network2.DTN()
    class_num = 10
    ad_net = network2.AdversarialNetwork(network.output_num() * class_num, 500)
    # print(network)
    # network = network2.resnet50(pretrained=True)
    num_epochs = 50
    lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.03, total_step=step_size * num_epochs,
                            step_per_epoch=step_size, decay_epoch=num_epochs)
    # 定义优化器和损失函数

    conv_params = list(filter(lambda x: "conv_params" in x.name, network.trainable_params()))
    fc_params = list(filter(lambda x: "fc_params" in x.name, network.trainable_params()))
    cls_params = list(filter(lambda x: "classifier" in x.name, network.trainable_params()))

    parameter_list_D1 = [
        {'params': ad_net.trainable_params()}
    ]

    parameter_list_G = [{'params': conv_params},
                        {'params': fc_params},
                        {'params': cls_params}
                        ]

    parameter_list_D2 = [{'params': conv_params},
                         {'params': fc_params}
                         ]

    opt_D1 = nn.SGD(params=parameter_list_D1, learning_rate=lr, weight_decay=0.0005, momentum=0.9)
    opt_G = nn.SGD(params=parameter_list_G, learning_rate=lr, weight_decay=0.0005, momentum=0.9)
    opt_D2 = nn.SGD(params=parameter_list_D2, learning_rate=lr, weight_decay=0.0005, momentum=0.9)
    # opt = nn.Momentum(params=network.trainable_params(), learning_rate=lr, momentum=0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False,reduction="mean")
    # # 实例化模型
    # model = ms.Model(network, loss, opt, metrics=None)
    # # 模型训练
    # model.train(num_epochs, source_dataset, callbacks=[LossMonitor()])  # callbacks=[ValAccMonitor(model, test_loader, num_epochs)])

    # 使用TrainOneStepCell自定义网络
    # 第一次训练
    loss_net_G = network2.CustomWithLossCell_G(network, loss)  # 包含损失函数的Cell
    train_net_G = network2.CustomTrainOneStepCell_G(loss_net_G, opt_G)
    # 第二次训练
    loss_net_D1 = network2.CustomWithLossCell_D1(network, loss, ad_net)
    train_net_D1 = network2.CustomTrainOneStepCell_D1(loss_net_D1, opt_D1)
    # 第三次训练
    loss_net_D2 = network2.CustomWithLossCell_D2(network, loss, ad_net)
    train_net_D2 = network2.CustomTrainOneStepCell_D2(loss_net_D2, opt_D2)

    for i in range(num_epochs):
        a_loss = 0
        b_loss = 0
        c_loss = 0
        # for image, label in source_dataset:
        len_train_source = source_dataset.get_dataset_size()
        len_train_target = source_dataset1.get_dataset_size()
        if len_train_source > len_train_target:
            num_iter = len_train_source
        else:
            num_iter = len_train_target
        for batch_idx in range(num_iter):
            if batch_idx % len_train_source == 0:
                iter_source = source_dataset.create_dict_iterator()
            if batch_idx % len_train_target == 0:
                iter_target = source_dataset1.create_dict_iterator()
            if batch_idx == 2000:
                print(2)
            batch_source = next(iter_source)
            batch_target = next(iter_target)
            inputs_source, inputs_target = batch_source['image'], batch_target['image']
            labels_source, labels_target = batch_source['label'], batch_target['label']

            # domain_source, domain_target = batch_source['domain'], batch_target['domain']
            image = ops.Concat()((inputs_source, inputs_target))
            image = Tensor(image, dtype=ms.float32)
            # label = Tensor(label, dtype=ms.float32)
            train_net_G.set_train()
            loss_num = train_net_G(image, labels_source, inputs_source)  # 执行网络的单步训练

            train_net_D1.set_train()
            ad_loss1 = train_net_D1(image, labels_source)

            train_net_D2.set_train()
            ad_loss2 = train_net_D2(image, labels_source)

            a = loss_num.item(0)
            a_loss += a


            b = ad_loss1.item(0)
            b_loss += b


            c = ad_loss2.item(0)
            c_loss += c

        print(i)

        print('Loss:', a_loss / num_iter)
        print('Loss:', b_loss / num_iter)
        print('Loss:', c_loss / num_iter)


    # plt.imshow(img)
    # plt.show()

        # -------------------------------------------------------------------------------------------------------------#
        # model = network.DTN()
        # model = model.cuda()
        test_loss = 0
        correct = 0
        len_test = test_dataset.get_dataset_size()
        num_iter = len_test
        iter_test = test_dataset.create_dict_iterator()
        for batch_idx in range(num_iter):

            batch_test = next(iter_test)

            inputs_test= batch_test['image']
            labels_test = batch_test['label']

            # domain_source, domain_target = batch_source['domain'], batch_target['domain']
            image = Tensor(inputs_test, dtype=ms.float32)
            # label = Tensor(label, dtype=ms.float32)

            # train_net_G.set_train()
            # loss_num = train_net_G(image, labels_source, inputs_source)  # 执行网络的单步训练
            feature, output = network(image)
            labels = nn.OneHot(depth=10)(labels_test)
            test_loss += nn.SoftmaxCrossEntropyWithLogits(sparse=False,reduction="mean")(output, labels)
            index, _ = ops.ArgMaxWithValue(axis=1)(output)
            num = index == labels_test
            for i in range(num.shape[0]):
                if num[i] == True :
                    correct += 1

        test_loss /= num_iter
        # a = test_loss.item(0)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss.asnumpy(), correct, num_iter*32,
            100.* correct / (num_iter*32)))



        print("end")


if __name__ == '__main__':
    main()
