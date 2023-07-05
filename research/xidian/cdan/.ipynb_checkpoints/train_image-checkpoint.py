import argparse
import wandb
import os.path as osp
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
import lr_schedule
# import loss as loss_func
import network2 as network
import network2
import new_trainer
from mindspore import Tensor
from mindvision.engine.callback import LossMonitor
import mindspore.dataset.vision as vision
from mindspore import context, Tensor, ParameterTuple
from mindspore import nn, Model, ops
from mindspore import dtype as mstype
import pre_process as prep
import warnings
warnings.filterwarnings("ignore")

# 动态图
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

# Define sweep config
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'test_acc'},
    'parameters': 
    {
        'momentum': {'max': 0.95, 'min': 0.7},
        'lr': {'max': 0.005, 'min': 0.0001}
     }
}
# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='cdan')


def inv_lr_scheduler(lr_mult, total_step, gamma, power, lr=0.001):
    lr_list = []
    lr_ori = lr
    for i in range(total_step):
        lr = lr_ori * (1 + gamma * i) ** (-power)
        lr = lr * lr_mult
        lr_list.append(lr)
    # print(lr_list)
    # print(len(lr_list))
    return lr_list

def create_dataset_imagenet(dataset_path, bs):
    """数据加载"""
    data_set = ds.ImageFolderDataset(dataset_path,
                                     num_parallel_workers=4,
                                     shuffle=True,
                                     decode=True)

    # 数据增强操作
    # transform_img = [
    #     ds.vision.c_transforms.Resize((32,32)),
    #     ds.vision.c_transforms.Normalize((0.5,), (0.5,)),
    #     ds.vision.c_transforms.CenterCrop((32,32)),
    #     ds.vision.c_transforms.HWC2CHW(),
    #     # lambda x: ((x / 255).astype("float32"), np.random.normal(size=(100, 1, 1)).astype("float32"))
    #     ]
    
    transform_img = [
        ds.vision.c_transforms.Resize((256,256)),
        ds.vision.c_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
        ds.vision.c_transforms.RandomResizedCrop((224,224)),
        ds.vision.c_transforms.HWC2CHW(),
        # lambda x: ((x / 255).astype("float32"), np.random.normal(size=(100, 1, 1)).astype("float32"))
        ]

    # 数据映射操作
    data_set = data_set.map(input_columns="image",
                            num_parallel_workers=4,
                            operations=transform_img)

    # 批量操作
    data_set = data_set.batch(bs, drop_remainder=True)
    return data_set

def create_dataset_imagenet1(dataset_path, bs):
    """数据加载"""
    data_set = ds.ImageFolderDataset(dataset_path,
                                     num_parallel_workers=4,
                                     shuffle=True,
                                     decode=True)

    transform_img = [
        ds.vision.c_transforms.Resize((256,256)),
        ds.vision.c_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
        ds.vision.c_transforms.CenterCrop((224,224)),
        ds.vision.c_transforms.HWC2CHW(),
        ]

    # 数据映射操作
    data_set = data_set.map(input_columns="image",
                            num_parallel_workers=4,
                            operations=transform_img)

    # 批量操作
    data_set = data_set.batch(bs, drop_remainder=True)
    return data_set

def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy



def train(configs, bs):
    ## set pre-process
    prep_dict = {}
    prep_config = configs["prep"]
    prep_dict["source"] = prep.image_train(**configs["prep"]['params'])
    prep_dict["target"] = prep.image_train(**configs["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**configs["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**configs["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = configs["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    
    # image_folder_dataset_dir = "../data/svhn2mnist/svhn_image"
    # source_dataset = create_dataset_imagenet(image_folder_dataset_dir)
    # dset_loaders["source"] = source_dataset
    
    dset_loaders["source"] = create_dataset_imagenet(data_config["source"]["list_path"], bs)
    dset_loaders["target"] = create_dataset_imagenet1(data_config["target"]["list_path"], bs)

    step_size = dset_loaders["source"].get_dataset_size()
    # # 获取数据集大小
    # step_size = dset_loaders["source"].get_dataset_size()
    # data_iter = next(dset_loaders["source"].create_dict_iterator(output_numpy=True))
    # # # 可视化部分训练数据
    # plt.figure(figsize=(10, 3), dpi=140)
    # for i, image in enumerate(data_iter['image'][:30], 1):
    #     plt.subplot(3, 10, i)
    #     plt.axis("off")
    #     image = image / image.max()
    #     plt.imshow(image.transpose(1, 2, 0))
    # plt.show()

    if prep_config["test_10crop"]:
        for i in range(10):
            dset_loaders["test"] = create_dataset_imagenet(data_config["test"]["list_path"], bs)
    else:
        dset_loaders["test"] = create_dataset_imagenet1(data_config["test"]["list_path"], bs)

    class_num = configs["network"]["params"]["class_num"]

    ## set base network
    net_config = configs["network"]
    base_network = net_config["name"](**net_config["params"])

    ## add additional network for some methods
    if configs["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], configs["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(configs["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)

    feature_layer_params = list(filter(lambda x: "back_bone" in x.name, base_network.trainable_params()))
    bottleneck_params = list(filter(lambda x: "bottleneck" in x.name, base_network.trainable_params()))
    fc_params = list(filter(lambda x: "fc" in x.name, base_network.trainable_params()))

    optimizer_config = configs['optimizer']
    schedule_param = optimizer_config['lr_param']
    
#     lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.03, total_step=step_size * (config['source_iters']),
#                            step_per_epoch=step_size, decay_epoch=config['source_iters'])
    
#     lr_t = nn.cosine_decay_lr(min_lr=0.0009, max_lr=0.005, total_step=step_size *  1000, step_per_epoch=step_size, decay_epoch= 1000)
    # parameter_list = [{'params': feature_layer_params,
    #                    'lr': inv_lr_scheduler(lr_mult=1, total_step=config['source_iters'], **schedule_param)},
    #                   {'params': bottleneck_params,
    #                    'lr': inv_lr_scheduler(lr_mult=10, total_step=config['source_iters'], **schedule_param)},
    #                   {'params': fc_params,
    #                    'lr': inv_lr_scheduler(lr_mult=10, total_step=config['source_iters'], **schedule_param)},
    #                   ]
    parameter_list = [{'params': feature_layer_params,
                       'lr': args.lr},
                      {'params': bottleneck_params,
                       'lr': args.lr},
                      {'params': fc_params,
                       'lr': args.lr},
                      ]
    
    parameter_list_G = [{'params': feature_layer_params,
                       'lr': args.lr},
                      {'params': bottleneck_params,
                       'lr': args.lr},
                      {'params': fc_params,
                       'lr': args.lr},
                      ]
    parameter_list_D1 = [
        {'params': ad_net.trainable_params(),'lr': args.lr}
    ]

    parameter_list_D2 = [{'params': feature_layer_params,
                       'lr': args.lr},
                      {'params': bottleneck_params,
                       'lr': args.lr}]
                         
    optimizer_config = configs['optimizer']
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))
    
    opt_G = optimizer_config['type'](parameter_list_G, **(optimizer_config['optim_params']))
    opt_D1 = optimizer_config['type'](parameter_list_D1, **(optimizer_config['optim_params']))
    opt_D2 = optimizer_config['type'](parameter_list_D2, **(optimizer_config['optim_params']))
    
    # optimizer = nn.SGD(params=parameter_list, weight_decay=0.0005, momentum=0.9)
    loss_1 = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction="mean")
    loss_2 = nn.BCELoss(reduction="mean")
    
    loss_net_pre = new_trainer.CustomWithLossCell_G(base_network, loss_1)  # 包含损失函数的Cell
    train_net_pre = new_trainer.CustomTrainOneStepCell_G(loss_net_pre, optimizer)
    
                ########  预训练  ###########
        
    len_train_source = dset_loaders['source'].get_dataset_size()
    best_acc_pre = 0
    for i in range(configs['source_iters']):
        # 设置网络为训练模式
        train_net_pre.set_train()
        pre_loss_t = 0
        num_iter = len_train_source
        for batch_idx in range(num_iter):
            if batch_idx % len_train_source == 0:
                ds_source = dset_loaders['source'].create_dict_iterator()
                # print("begin data label is {}".format(next(ds_source)["target"]))
            batch_source = next(ds_source)
            image = batch_source['image']
            label = batch_source['label']
            pre_loss = train_net_pre(image, label,image)
            pre_loss_t += pre_loss
        print(i)
        print(pre_loss_t / num_iter)

    # -------------------------------------------------------------------------------------------------------------#
# model = network.DTN()
# model = model.cuda()
    test_loss = 0
    correct = 0
    len_test = len_train_source
    num_iter = len_test
    ds_source = dset_loaders['source'].create_dict_iterator()
    for batch_idx in range(num_iter):
        batch_test = next(ds_source)
        inputs_test = batch_test['image']
        labels_test = batch_test['label']

          # 执行网络的单步训练
        _ , output = base_network(inputs_test)
        labels = nn.OneHot(depth=31)(labels_test)
        test_loss += nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction="mean")(output, labels)
        index, _ = ops.ArgMaxWithValue(axis=1)(output)
        num = index == labels_test
        for i in range(num.shape[0]):
            if num[i] == True:
                correct += 1

    test_loss /= num_iter
    # a = test_loss.item(0)

    acc = 100. * correct / (num_iter * bs)

    if best_acc_pre < acc:
        best_acc_pre = acc

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss.asnumpy(), correct, num_iter*bs, acc))




    print('Best Accuracy:{:.0f}%'.format(best_acc_pre))
    print("end")
    
    ####### 训练阶段 #######
    # 第一次训练
    loss_net_pre = new_trainer.CustomWithLossCell_G(base_network, loss_1)  # 包含损失函数的Cell
    train_net_pre = new_trainer.CustomTrainOneStepCell_G(loss_net_pre, opt_G)
    # 第二次训练
    loss_net_D1 = network2.CustomWithLossCell_D1(base_network, loss_2, ad_net, random_layer)
    train_net_D1 = network2.CustomTrainOneStepCell_D1(loss_net_D1, opt_D1)
    # 第三次训练
    loss_net_D2 = network2.CustomWithLossCell_D2(base_network, loss_2, ad_net, random_layer)
    train_net_D2 = network2.CustomTrainOneStepCell_D2(loss_net_D2, opt_D2)
    best_acc = 0
    for i in range(1000):
        a_loss = 0
        b_loss = 0
        c_loss = 0
        # for image, label in source_dataset:
        len_train_source = dset_loaders['source'].get_dataset_size()
        len_train_target = dset_loaders['target'].get_dataset_size()
        if len_train_source > len_train_target:
            num_iter = len_train_source
        else:
            num_iter = len_train_target
        for batch_idx in range(num_iter):
            if batch_idx % len_train_source == 0:
                iter_source = dset_loaders['source'].create_dict_iterator()
            if batch_idx % len_train_target == 0:
                iter_target = dset_loaders['target'].create_dict_iterator()
            batch_source = next(iter_source)
            batch_target = next(iter_target)
            inputs_source, inputs_target = batch_source['image'], batch_target['image']
            labels_source, labels_target = batch_source['label'], batch_target['label']

            # domain_source, domain_target = batch_source['domain'], batch_target['domain']
            image = ops.Concat()((inputs_source, inputs_target))
            image = Tensor(image, dtype=ms.float32)
            # label = Tensor(label, dtype=ms.float32)
            train_net_pre.set_train()
            loss_num = train_net_pre(image, labels_source, inputs_source)  # 执行网络的单步训练

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
        len_test = dset_loaders['target'].get_dataset_size()
        num_iter = len_test
        iter_test = dset_loaders['target'].create_dict_iterator()
        for batch_idx in range(num_iter):

            batch_test = next(iter_test)

            inputs_test= batch_test['image']
            labels_test = batch_test['label']

            # domain_source, domain_target = batch_source['domain'], batch_target['domain']
            image = Tensor(inputs_test, dtype=ms.float32)
            # label = Tensor(label, dtype=ms.float32)

            # train_net_G.set_train()
            # loss_num = train_net_G(image, labels_source, inputs_source)  # 执行网络的单步训练
            feature, output = base_network(image)
            labels = nn.OneHot(depth=31)(labels_test)
            test_loss += nn.SoftmaxCrossEntropyWithLogits(sparse=False,reduction="mean")(output, labels)
            index, _ = ops.ArgMaxWithValue(axis=1)(output)
            num = index == labels_test
            for i in range(num.shape[0]):
                if num[i] == True :
                    correct += 1

        test_loss /= num_iter
        # a = test_loss.item(0)
        
        acc = 100. * correct / (num_iter * bs)
    
        if best_acc < acc:
            best_acc = acc
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss.asnumpy(), correct, num_iter*bs, acc))



        
        print('Best Accuracy:{:.0f}%'.format(best_acc))
        print("end")
    
    return best_acc_pre, best_acc

    '''''
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        

    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, \
                base_network, test_10crop=prep_config["test_10crop"])
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)
        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                "iter_{:05d}_model.pth.tar".format(i)))

        loss_params = config["loss"]                  
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        if config['method'] == 'CDAN+E':           
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
        elif config['method']  == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
        elif config['method']  == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../data/office/domain_adaptation_images/webcam/images', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../data/office/domain_adaptation_images/amazon/images', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=2000, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="learning rate")
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--source_iters', type=int, default=80, help='number of source pre-train iters')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # train configs
    configs = {}
    configs['method'] = args.method
    configs["gpu"] = args.gpu_id
    configs["num_iterations"] = 100004
    configs['source_iters'] = args.source_iters
    configs["test_interval"] = args.test_interval
    configs["snapshot_interval"] = args.snapshot_interval
    configs["output_for_test"] = True
    configs["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(configs["output_path"]):
        os.system('mkdir -p '+configs["output_path"])
    configs["out_file"] = open(osp.join(configs["output_path"], "log.txt"), "w")
    if not osp.exists(configs["output_path"]):
        os.mkdir(configs["output_path"])

    configs["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    configs["loss"] = {"trade_off":1.0}
    if "AlexNet" in args.net:
        configs["prep"]['params']['alexnet'] = True
        configs["prep"]['params']['crop_size'] = 227
        configs["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        configs["network"] = {"name":network.ResNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        configs["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    configs["loss"]["random"] = args.random
    configs["loss"]["random_dim"] = 1024

    # config["optimizer"] = {"type":nn.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
    #                        "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
    #                        "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    configs["dataset"] = args.dset
    configs["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":4}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":4}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":4}}
    
    configs['optimizer'] = {
        'type': nn.SGD,
        'optim_params': {
            # 'learning_rate': args.lr,
            'momentum': args.momentum,
            'weight_decay': args.wd * 2,  # 源代码中，decay始终x2，所以这里保持和源代码一致
            'nesterov': True,
        },
        'lr_param': {
            'lr': args.lr,
            'gamma': 0.001,
            'power': 0.75,
        },
    }

    if configs["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            configs["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            configs["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters       
        configs["network"]["params"]["class_num"] = 31 
    elif configs["dataset"] == "image-clef":
        configs["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        configs["network"]["params"]["class_num"] = 12
    elif configs["dataset"] == "visda":
        configs["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        configs["network"]["params"]["class_num"] = 12
        configs['loss']["trade_off"] = 1.0
    elif configs["dataset"] == "office-home":
        configs["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        configs["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    configs["out_file"].write(str(configs))
    configs["out_file"].flush()
    
    
def main() :
    wandb.init(
    # set the wandb project where this run will be logged
    project="cdan",
    
    # track hyperparameters and run metadata
    config={
    "lr": 0.01,
    "batch_size": 64,
    "momentum":0.9,
    })
    
    args.momentum = wandb.config.momentum
    args.lr =  wandb.config.lr
    bs = wandb.config.batch_size
    
    pre_train_acc, test_acc = train(configs, bs)
    
    wandb.log({
        'pre_train_acc': pre_train_acc,
        'test_acc': test_acc
      })
    
    
    
    
    
# Start sweep job.
wandb.agent(sweep_id, function=main, count=5)
    
    
