from __future__ import print_function
import os
import sys
import time
import argparse
import numpy as np

# 自定义模块导入
from ChannelAug import ChannelRandomErasing
from data_loader_ms import VideoDataset_train, VideoDataset_test
from data_manager import VCM
from eval_metrics import evaluate
from loss import OriTripletLoss
from model_ms import EmbedNet as embed_net
from utils import *

# MindSpore相关模块导入
from mindspore import context, nn, Tensor
from mindspore.train.callback import SummaryCollector
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import (
    ToPIL,
    Resize,
    Pad,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize
)


parser = argparse.ArgumentParser(description='Mindspore Cross-Modality Training')
parser.add_argument('--dataset', default='VCM', help='dataset name: VCM(Video Cross-modal)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--exp_name', default='test', type=str, help='exp name')
parser.add_argument('--gml', default=False, type=bool, help='global mutual learning')
parser.add_argument('--lml', default=False, type=bool, help='local mutual learning')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='autodl-fs/checkpoint/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log_vcm/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=32, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--part', default=3, type=int,
                    metavar='tb', help=' part number')
parser.add_argument('--method', default='id+tri', type=str,
                    metavar='m', help='method type')
parser.add_argument('--drop', default=0.2, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=2, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--lambda0', default=1.0, type=float,
                    metavar='lambda0', help='graph attention weights')
parser.add_argument('--graph', action='store_true', help='either add graph attention or not')
parser.add_argument('--wpa', action='store_true', help='either add weighted part attention')
parser.add_argument('--a', default=1, type=float,
                    metavar='lambda1', help='dropout ratio')
args = parser.parse_args()

# 设置设备
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)

dataset = args.dataset
seq_lenth = 12
test_batch = 32
data_set = VCM()

# 获取当前时间
current_time = time.localtime()
cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

# 参数设置
test_mode = [1, 2]
height = args.img_h
width = args.img_w
global_mutual_learning = args.gml
local_mutual_learning = args.lml
warm_up_epochs = 0

if global_mutual_learning and local_mutual_learning:
    log_path = args.log_path + cur_time + '_G_L_VCM_log_' + args.exp_name + '/'
elif global_mutual_learning:
    log_path = args.log_path + cur_time + '_G_VCM_log_' + args.exp_name + '/'
else:
    log_path = args.log_path + cur_time + '_base_VCM_log_' + args.exp_name + '/'

checkpoint_path = args.model_path + cur_time + args.exp_name + "/"

os.makedirs(log_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(args.vis_log_path, exist_ok=True)

# 日志文件路径
suffix = f"_{dataset}_lr_{args.lr}_batchsize_{args.batch_size}"
if args.optim != 'sgd':
    suffix += f"_{args.optim}"

test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path + suffix + '_os.txt')

# 可视化日志路径
vis_log_dir = args.vis_log_path + cur_time + suffix + "_" + args.exp_name + '/'
os.makedirs(vis_log_dir, exist_ok=True)

summary_collector = SummaryCollector(log_dir=vis_log_dir)

print(f"==========\nArgs: {args}\n==========")
best_acc = 0
best_acc_v2t = 0
best_map_acc = 0
best_map_acc_v2t = 0

start_epoch = 0
feature_dim = args.low_dim

print('==> Loading data...')

# 修复后的数据预处理
transform_train = Compose([
    ToPIL(),  # 将图像转换为 PIL 格式
    Resize((288, 144)),  # 调整大小
    Pad(10),  # 填充边界
    RandomCrop((args.img_h, args.img_w)),  # 随机裁剪
    RandomHorizontalFlip(prob=0.5),  # 随机水平翻转，添加 prob 参数
    ToTensor(),  # 转换为张量
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

transform_test = Compose([
    ToPIL(),
    Resize((args.img_h, args.img_w)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


if dataset == 'VCM':
    rgb_pos, ir_pos = GenIdx(data_set.rgb_label, data_set.ir_label)

queryloader = VideoDataset_test(data_set.query, seq_len=seq_lenth, sample='video_test', transform=transform_test).create_tuple_iterator(batch_size=test_batch, shuffle=False)
galleryloader = VideoDataset_test(data_set.gallery, seq_len=seq_lenth, sample='video_test', transform=transform_test).create_tuple_iterator(batch_size=test_batch, shuffle=False)

# 可见光到红外的测试集
queryloader_1 = VideoDataset_test(data_set.query_1, seq_len=seq_lenth, sample='video_test', transform=transform_test).create_tuple_iterator(batch_size=test_batch, shuffle=False)
galleryloader_1 = VideoDataset_test(data_set.gallery_1, seq_len=seq_lenth, sample='video_test', transform=transform_test).create_tuple_iterator(batch_size=test_batch, shuffle=False)

nquery_1 = data_set.num_query_tracklets_1
ngall_1 = data_set.num_gallery_tracklets_1

n_class = data_set.num_train_pids
nquery = data_set.num_query_tracklets
ngall = data_set.num_gallery_tracklets

print('==> Building model...')
net = embed_net(class_num=n_class)

if args.resume:
    model_path = args.resume
    if os.path.isfile(model_path):
        print(f"==> Loading checkpoint {args.resume}")
        param_dict = load_checkpoint(model_path)
        load_param_into_net(net, param_dict)
        print(f"==> Loaded checkpoint {args.resume}")

# 定义损失函数
criterion1 = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
criterion2 = OriTripletLoss(batch_size=args.batch_size * args.num_pos, margin=args.margin)
criterion3 = nn.KLDivLoss()

# 定义优化器
if args.optim == 'sgd':
    optimizer = nn.SGD(net.trainable_params(), learning_rate=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
elif args.optim == 'adam':
    optimizer = nn.Adam(net.trainable_params(), learning_rate=args.lr, weight_decay=5e-4)

def adjust_learning_rate(optimizer_P, epoch):
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 60:
        lr = args.lr
    elif 60 <= epoch < 120:
        lr = args.lr * 0.1
    elif epoch >= 120:
        lr = args.lr * 0.01

    # cur_lr = optimizer_P.param_groups[0]['lr']
    optimizer_P.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer_P.param_groups) - 1):
        optimizer_P.param_groups[i + 1]['lr'] = lr
    # optimizer_P.param_groups[0]['lr'] = lr

    return lr


def train(epoch, wG):
    # Adjust learning rate
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    lid_loss = AverageMeter()
    did_loss = AverageMeter()
    tri_loss = AverageMeter()
    de_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    net.set_train(True)  # MindSpore 设置为训练模式
    end = time.time()

    for batch_idx, (imgs_ir, imgs_ir_p, pids_ir, camid_ir, imgs_rgb, imgs_rgb_p, pids_rgb, camid_rgb) in enumerate(trainloader):

        # 输入数据与标签
        input1 = Tensor(imgs_rgb, dtype=mindspore.float32)
        input2 = Tensor(imgs_ir, dtype=mindspore.float32)
        input3 = Tensor(imgs_rgb_p, dtype=mindspore.float32)
        input4 = Tensor(imgs_ir_p, dtype=mindspore.float32)

        label1 = Tensor(pids_rgb, dtype=mindspore.int32)
        label2 = Tensor(pids_ir, dtype=mindspore.int32)
        labels = ops.concat((label1, label2), axis=0)

        data_time.update(time.time() - end)

        # 前向传播
        feat, x_local, logits, l_logits, l_d_logits, loss_defense = net(input1, input2, input3, input4, seq_len=seq_lenth)

        # 计算损失
        loss_id = criterion1(logits, labels)
        loss_tri, batch_acc = criterion2(feat, labels)  # Triplet loss
        correct += (batch_acc / 2)

        predicted = ops.ArgMaxWithValue(axis=1)(logits)[0]
        correct += (ops.equal(predicted, labels).astype(mindspore.float32).sum().asnumpy() / 2)

        local_label_1 = ops.concat([label1.expand_dims(1)] * seq_lenth, axis=1).reshape(-1)
        local_label_2 = ops.concat([label2.expand_dims(1)] * seq_lenth, axis=1).reshape(-1)
        local_label = ops.concat((local_label_1, local_label_2), axis=0)

        l_id_loss = criterion1(l_logits, local_label)
        d_id_loss = criterion1(l_d_logits, local_label)

        loss = loss_id + loss_tri + l_id_loss + d_id_loss + loss_defense

        # 优化器清零
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新损失与准确率
        train_loss.update(loss.asnumpy(), 2 * input1.shape[0])
        id_loss.update(loss_id.asnumpy(), 2 * input1.shape[0])
        lid_loss.update(l_id_loss.asnumpy(), 2 * input1.shape[0])
        did_loss.update(d_id_loss.asnumpy(), 2 * input1.shape[0])
        tri_loss.update(loss_tri.asnumpy(), 2 * input1.shape[0])
        de_loss.update(loss_defense.asnumpy(), 2 * input1.shape[0])

        total += labels.shape[0]

        # 计算时间
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10 == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] '
                  f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'lr: {current_lr} '
                  f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  f'i Loss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  f'li Loss: {lid_loss.val:.4f} ({lid_loss.avg:.4f}) '
                  f'di Loss: {did_loss.val:.4f} ({did_loss.avg:.4f}) '
                  f't Loss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  f'de Loss: {de_loss.val:.4f} ({de_loss.avg:.4f}) '
                  f'Accu: {100. * correct / total:.2f}')

    print(f"[Epoch {epoch}] lr: {current_lr}, Avg Loss: {train_loss.avg:.4f}, "
          f"Loss_id: {id_loss.avg:.4f}, lid_loss: {lid_loss.avg:.4f}, did_loss: {did_loss.avg:.4f}, "
          f"Loss_tri: {tri_loss.avg:.4f}, de_loss: {de_loss.avg:.4f}, "
          f"Batch Time: {batch_time.avg:.3f}, Accuracy: {100. * correct / total:.2f}%")

    return 1. / (1. + train_loss.avg)


def test2(epoch):
    # 切换到评估模式
    net.set_train(False)
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall_1, 2048))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []

    for batch_idx, (imgs, imgs_ir_p, pids, camids) in enumerate(galleryloader_1):
        input = Tensor(imgs, dtype=mindspore.float32)
        input_pose = Tensor(imgs_ir_p, dtype=mindspore.float32)
        batch_num = input.shape[0]

        feat = net(input, input, input_pose, input_pose, test_mode[1], seq_len=seq_lenth)
        gall_feat[ptr:ptr + batch_num, :] = feat.asnumpy()
        ptr = ptr + batch_num

        g_pids.extend(pids)
        g_camids.extend(camids)

    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery_1, 2048))

    for batch_idx, (imgs, imgs_ir_p, pids, camids) in enumerate(queryloader_1):
        input = Tensor(imgs, dtype=mindspore.float32)
        input_pose = Tensor(imgs_ir_p, dtype=mindspore.float32)
        batch_num = input.shape[0]

        feat = net(input, input, input_pose, input_pose, test_mode[0], seq_len=seq_lenth)
        query_feat[ptr:ptr + batch_num, :] = feat.asnumpy()
        ptr = ptr + batch_num

        q_pids.extend(pids)
        q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # 计算相似性
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # 评估
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc, mAP


def test(epoch):
    # 切换到评估模式
    net.set_train(False)
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []

    for batch_idx, (imgs, imgs_ir_p, pids, camids) in enumerate(galleryloader):
        input = Tensor(imgs, dtype=mindspore.float32)
        input_pose = Tensor(imgs_ir_p, dtype=mindspore.float32)
        batch_num = input.shape[0]

        feat = net(input, input, input_pose, input_pose, test_mode[0], seq_len=seq_lenth)
        gall_feat[ptr:ptr + batch_num, :] = feat.asnumpy()
        ptr = ptr + batch_num

        g_pids.extend(pids)
        g_camids.extend(camids)

    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))

    for batch_idx, (imgs, imgs_ir_p, pids, camids) in enumerate(queryloader):
        input = Tensor(imgs, dtype=mindspore.float32)
        input_pose = Tensor(imgs_ir_p, dtype=mindspore.float32)
        batch_num = input.shape[0]

        feat = net(input, input, input_pose, input_pose, test_mode[1], seq_len=seq_lenth)
        query_feat[ptr:ptr + batch_num, :] = feat.asnumpy()
        ptr = ptr + batch_num

        q_pids.extend(pids)
        q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # 计算相似性
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)

    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc, mAP





print('==> Start Training...')

for epoch in range(start_epoch, 201 - start_epoch):

    print('==> Preparing Data Loader...')
    # IdentitySampler
    sampler = IdentitySampler(data_set.ir_label, data_set.rgb_label, rgb_pos, ir_pos, args.num_pos, args.batch_size)

    index1 = sampler.index1  # ndarray, 所有 RGB 模态轨迹
    index2 = sampler.index2  # ndarray, 所有 IR 模态轨迹

    loader_batch = args.batch_size * args.num_pos

    # DataLoader 转换为 MindSpore 数据集
    train_dataset = VideoDataset_train(
        data_set.train_ir, data_set.train_rgb, seq_len=seq_lenth,
        sample='video_train', transform=transform_train, index1=index1, index2=index2
    )
    trainloader = GeneratorDataset(train_dataset, ['imgs_ir', 'imgs_ir_p', 'pids_ir', 'camid_ir', 
                                                   'imgs_rgb', 'imgs_rgb_p', 'pids_rgb', 'camid_rgb'],
                                   sampler=sampler)
    trainloader = trainloader.batch(loader_batch, drop_remainder=True)

    # 开始训练
    wG = train(epoch, wG)

    if epoch % 10 == 0:
        print('Test Epoch: {}'.format(epoch))
        print('Test Epoch: {}'.format(epoch), file=test_log_file)

        # Testing
        cmc, mAP = test(epoch)

        if cmc[0] > best_acc:
            best_acc = cmc[0]
            best_epoch = epoch
            state = {
                'net': net.parameters_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
            save_checkpoint(state, checkpoint_path + suffix + 't2v_rank1_best.ckpt')

        if mAP > best_map_acc:
            best_map_acc = mAP
            best_epoch = epoch
            state = {
                'net': net.parameters_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
            save_checkpoint(state, checkpoint_path + suffix + 't2v_map_best.ckpt')

        print(
            'FC(t2v):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        print('Best t2v epoch [{}]'.format(best_epoch))
        print(
            'FC(t2v):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=test_log_file)
        print(f"[Epoch {epoch}] t2v-Rank-1: {cmc[0]:.2%}, t2v-Rank-5: {cmc[4]:.2%}, "
              f"t2v-Rank-10: {cmc[9]:.2%}, t2v-Rank-20: {cmc[19]:.2%}, mAP_t2v: {mAP:.2%}")

        # 测试第二部分
        cmc, mAP = test2(epoch)
        if cmc[0] > best_acc_v2t:
            best_acc_v2t = cmc[0]
            best_epoch = epoch
            state = {
                'net': net.parameters_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
            save_checkpoint(state, checkpoint_path + suffix + 'v2t_rank1_best.ckpt')

        if mAP > best_map_acc_v2t:
            best_map_acc_v2t = mAP
            best_epoch = epoch
            state = {
                'net': net.parameters_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
            save_checkpoint(state, checkpoint_path + suffix + 'v2t_map_best.ckpt')

        print(
            'FC(v2t):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        print('Best v2t epoch [{}]'.format(best_epoch))
        print(
            'FC(v2t):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=test_log_file)
        print(f"[Epoch {epoch}] t2v-Rank-1: {cmc[0]:.2%}, t2v-Rank-5: {cmc[4]:.2%}, "
              f"t2v-Rank-10: {cmc[9]:.2%}, t2v-Rank-20: {cmc[19]:.2%}, mAP_t2v: {mAP:.2%}")
        test_log_file.flush()
