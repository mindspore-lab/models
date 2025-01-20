import argparse
import os

import mindspore as ms
import numpy as np
import datetime

from base.base_train import Train as Train_pynative
from base.base_train_graph import Train as Train_graph
from utils.logger import MyLog
from utils.args2json import parse_args, print_options


def parseArgs():
    parser = argparse.ArgumentParser(description='SSDA Classification')
    # mindspore env setting
    parser.add_argument('--mode', type=str, default='GRAPH', help='mindsore mode-[PYNATIVE GRAPH]')
    parser.add_argument('--device-target', type=str, default='Ascend', help='device type-[Ascend GPU]')
    parser.add_argument('--device-id', type=int, default=0, help='device id')
    parser.add_argument('--max-device-memory', type=str, default='8GB', help='max-device-memory')
    parser.add_argument('--mempool-block-size', type=str, default='8GB', help='max-block-size')

    # source target datasets setting
    parser.add_argument('--source', type=str, default='real', help='source domain')
    parser.add_argument('--target', type=str, default='sketch', help='target domain')
    parser.add_argument('--dataset', type=str, default='multi', choices=['multi', 'office', 'office_home'], help='the name of dataset')
    parser.add_argument('--num', type=int, default=3, help='number of labeled examples in the target')

    parser.add_argument('--resume', action='store_true', default=False, help='resume to train')
    parser.add_argument('--test', action='store_true', default=False, help='test or not')

    # image and txt file path
    parser.add_argument('--txt-path', type=str, default='', help='split images txt path')
    parser.add_argument('--img-root', type=str, default='', help='img root')

    # algorithm select
    parser.add_argument('--method', type=str, default='MME', choices=['S+T', 'ENT', 'MME'], help='MME is proposed method, ENT is entropy minimization, S+T is training only on labeled examples')

    # optim and network setting
    parser.add_argument('--net', type=str, default='resnet34', choices=['alexnet', 'vgg16', 'resnet34'], help='which network to use')
    parser.add_argument('--steps', type=int, default=10000, metavar='N', help='maximum number of iterations to train (default: 50000)')
    parser.add_argument('--lr-init', type=float, default=0.01, metavar='LR', help='initial learning rate (default: 0.001)')
    parser.add_argument('--multi', type=float, default=0.1, metavar='MLT', help='learning rate multiplication')
    parser.add_argument('--T', type=float, default=0.05, metavar='T', help='temperature (default: 0.05)')
    parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM', help='value of lamda')
    parser.add_argument('--save_check', action='store_true', default=False, help='save checkpoint or not')

    #存权重,日志，配置文件的路径
    parser.add_argument('--save-path', type=str, default='./log', help='dir to save checkpoint')
    parser.add_argument('--log-name', type=str, default='train', help='dir to save checkpoint')
    parser.add_argument('--breakPoint', type=int, default=0, help='retrain from breakPoint')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--val_interval', type=int, default=500, metavar='N', help='how many batches to wait before saving a model')


    datenow = datetime.datetime.now()
    time_now_str = (
        str(datenow.year)
        + "-"
        + "{:02d}".format(datenow.month)
        + "{:02d}".format(datenow.day)
        + "-"
        + "{:02d}".format(datenow.hour)
        + "-"
        + "{:02d}".format(datenow.minute)
    )

    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path, args.dataset, args.source, args.target, time_now_str) 
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.txt_path = './data/txt/{}/'.format(args.dataset)
    args.img_root = './data/{}'.format(args.dataset)
    args = parse_args(args)
    return args

def main(args):
    if args['mode'] == 'PYNATIVE':
        mode = ms.PYNATIVE_MODE
    else:
        mode = ms.GRAPH_MODE
    ms.set_context(mode=mode, 
                device_target=args['device_target'], 
                device_id=args['device_id'], 
                max_device_memory=args['max_device_memory'], 
                mempool_block_size =args['mempool_block_size'])
    mylog = MyLog(log_dir=args['save_path'], logName=args['log_name']).logger
    mylog.info('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
        (args['dataset'], args['source'], args['target'], args['num'], args['net']))
    print_options(args, mylog)
    if mode == ms.PYNATIVE_MODE:
        T = Train_pynative(args, mylog)
    else:
        T = Train_graph(args, mylog)
    T.train()

if __name__ == '__main__':
    args = parseArgs()
    main(args)