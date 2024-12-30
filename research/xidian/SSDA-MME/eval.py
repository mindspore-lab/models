import argparse

import mindspore as ms
import numpy as np

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
    parser.add_argument('--resume', action='store_true', default=False, help='save checkpoint or not')

    parser.add_argument('--test', action='store_false', default=True, help='save checkpoint or not')
    parser.add_argument('--save-path', type=str, default='', help='weight log and config save path')

    args = parser.parse_args()
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
    mylog = MyLog(log_dir=args['save_path'], logName='Test').logger
    mylog.info('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
        (args['dataset'], args['source'], args['target'], args['num'], args['net']))
    print_options(args, mylog)
    if mode == ms.PYNATIVE_MODE:
        T = Train_pynative(args, mylog)
    else:
        T = Train_graph(args, mylog)
    T.load_model()
    T.test()

if __name__ == '__main__':
    args = parseArgs()
    main(args)