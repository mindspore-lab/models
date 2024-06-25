import numpy
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from src.convmlp import *
torch.manual_seed(114514)
torch.cuda.manual_seed(114514)    # reproducible
from utils.utils import *
# from utils.imgaugment import *
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import argparse

parser = argparse.ArgumentParser(description='Train CMLPNet')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch_size', type=int, default=128)
parser.add_argument('--dataset_choose', type=str, default='cifar10', help="cifar10 or cifar100")
parser.add_argument('--device_target', type=str, default='Ascend')
parser.add_argument('--save_checkpoint_path',
                    type=str,
                    default="./ckpt",
                    help='if is test, must provide\
                    path where the trained ckpt file')
args = parser.parse_args()

def acc_classes(pre, labels,BATCH_SIZE):
    pre_y = torch.max(pre, dim=1)[1]
    train_acc = torch.eq(pre_y, labels.to(device)).sum().item() / BATCH_SIZE
    return train_acc

def train(train_loader, model, criterion, optimizer, epoch, epoch_max,batchsize):
    """Train for one epoch on the training set"""
    losses_class = AverageMeter()
    acc = AverageMeter()
    runtime=0
    # switch to train mode
    model.train()
    with tqdm(total=len(train_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3) as pbar:
        for i, (input ,target) in enumerate(train_loader):
            start=time.time()
            images,labels = input,  target

            output= model(images.to(device))
            target_var=labels.to(device)
            # target_var = target_var.to(torch.float)

            loss = criterion(output, target_var)

            # measure accuracy and record loss
            acc.update(acc_classes(output.data, target,batchsize))

            losses_class.update(loss.item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end=time.time()
            runtime+=end-start
            pbar.set_postfix(**{'train_loss_': losses_class.avg,
                                'acc': acc.avg,'runtime':runtime/(i+1)})
            pbar.update(1)


    return acc.avg, losses_class.avg

def trainmixup(train_loader, model, criterion, optimizer, epoch, epoch_max,batchsize,alpha):
    """Train for one epoch on the training set"""
    losses_class = AverageMeter()
    acc = AverageMeter()
    # switch to train mode
    model.train()
    with tqdm(total=len(train_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3) as pbar:
        for i, (input ,target) in enumerate(train_loader):
            images,labels = input,  target
            images = images.cuda(non_blocking=True)
            labels = torch.from_numpy(np.array(labels)).float().cuda(non_blocking=True)
            # 2.mixup
            alpha = alpha
            lam = np.random.beta(alpha, alpha)
            # randperm返回1~images.size(0)的一个随机排列
            index = torch.randperm(images.size(0)).cuda()
            inputs = lam * images + (1 - lam) * images[index, :]
            targets_a, targets_b = labels, labels[index]

            output= model(inputs.to(device))
            target_var=labels.to(device)
            # target_var = target_var.to(torch.float)
            loss =  lam * criterion(output, targets_a.long()) + (1 - lam) * criterion(output, targets_b.long())

            # measure accuracy and record loss
            acc.update(acc_classes(output.data, target,batchsize))

            losses_class.update(loss.item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'train_loss_': losses_class.avg,
                                'acc': acc.avg})
            pbar.update(1)


    return acc.avg, losses_class.avg

def validate(val_loader, model, criterion, epoch, epoch_max,batchsize):
    """Perform validation on the validation set"""
    losses_class = AverageMeter()
    acc = AverageMeter()
    runtime = 0
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3,
                  colour='blue') as pbar:
            for i, (input,target) in enumerate(val_loader):
                start = time.time()
                val_image, val_label = input, target
                output = model(val_image.to(device))

                target_var = val_label.to(device)
                # target_var = target_var.to(torch.float)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                acc.update(acc_classes(output.data, target, batchsize))


                losses_class.update(loss.item())
                end = time.time()
                runtime += end - start
                pbar.set_postfix(**{'val_loss_class': losses_class.avg,
                                    'acc': acc.avg,'runtime':runtime/(i+1)})
                pbar.update(1)
    return acc.avg, losses_class.avg

if __name__ == '__main__':
    cifar_norm_mean = (0.4914, 0.4822, 0.4465)
    cifar_norm_std = (0.2470, 0.2435, 0.2616)
    train_trans= transforms.Compose(
        [transforms.Resize(224),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize(cifar_norm_mean, cifar_norm_std)])
    train_trans1 = transforms.Compose(
        [transforms.Resize(224),
         # transforms.RandomHorizontalFlip(p=0.5),
         # transforms.RandomVerticalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize(cifar_norm_mean, cifar_norm_std)])
    val_trans = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(cifar_norm_mean, cifar_norm_std)])

    batch_size = args.batch_size  # 每一个batch的大小
    num_epochs = args.num_epochs  # 将num_epochs提高到10
    lr = 0.0002  # 学习率
    wait = 10
    patience = 10
    alpha=0.8
    trainset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=train_trans,
    )
    trainset1 = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=train_trans1,
    )
    testset =torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=val_trans,

    )
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True,num_workers=0)
    train_loader1 = torch.utils.data.DataLoader(trainset1, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# GPU设置
    # num_classes = 10    # 手写数字识别，总共有10个类别的数字

    model=ConvMLP(blocks=[2, 4, 2], dims=[128, 256, 512], mlp_ratios=[2, 2, 2],
                        classifier_head=True, channels=64, n_conv_blocks=2,num_classes=10)
    # model=convmlp_s(pretrained=True,progress=True)
    # model.head = nn.Linear( 512, 10)
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.cuda()
    state_dict = torch.load(
        './checkpoint/cifa10_CMLP.pth')
    model.load_state_dict(state_dict)
    #-------------
    # model_dict = model.state_dict()
    # model_path='./checkpoint/cifa_best_network_acc_0.9548277243589743.pth'
    # pretrained_dict = torch.load(model_path, map_location=device)
    # load_key, no_load_key, temp_dict = [], [], {}
    # for k, v in pretrained_dict.items():
    #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
    #         temp_dict[k] = v
    #         load_key.append(k)
    # else:
    #     no_load_key.append(k)
    # model_dict.update(temp_dict)
    # model.load_state_dict(model_dict)
    # -------------
    early_stopping = EarlyStopping_acc(save_path='./checkpoint', patience=patience, wait=wait,
                                   choose='cifa',best_score=0.0)

    # dropout训练，训练阶段开启随机采样，所有模型共享参数
    # model.train()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() #选择损失函数
    optimizer1 = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.005,
        nesterov=True
    )
    optimizer2 = torch.optim.AdamW( model.parameters(), lr=lr, weight_decay=0.05,)
    optimizer=optimizer1
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,eta_min=1e-5)
    csv_logger = CSVStats()
    wait_idem=wait
    declay_count = 0
    # 训练模型
    for epoch in range(0, num_epochs):
        acc_train, loss_train = train(
            train_loader, model, criterion, optimizer, epoch, batchsize=batch_size, epoch_max=num_epochs)

        torch.cuda.empty_cache()

        acc_trainmix, loss_trainmix = trainmixup(
            train_loader1, model, criterion, optimizer, epoch, batchsize=batch_size, epoch_max=num_epochs,alpha=alpha)

        torch.cuda.empty_cache()
        acc_val, loss_val = validate(
            test_loader, model, criterion, epoch, batchsize=batch_size, epoch_max=num_epochs)
        lr_scheduler.step()
        # Print some statistics inside CSV
        csv_logger.add(acc_train, acc_val, loss_train, loss_val,lr_scheduler.get_lr()[0])
        csv_logger.write(patience=patience,wait=wait,choose='mlps',name='cifa')
        early_stopping(acc_val, model)
        # if early_stopping.flag == True:
        #     wait_idem = wait
        # if early_stopping.counter > 10:
        #     wait_idem += 1
        #     if wait_idem >= wait:
        #         lr = adjust_learning_rate(optimizer, lr, 0.5)
        #         wait_idem = 0
        #         declay_count += 1
        #     if declay_count >= 2:
        #         lr = adjust_learning_rate(optimizer, 0.001 * (0.5) ** 3, 0.5)
        #         declay_count = 0
        # print(wait)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break