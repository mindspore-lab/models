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

# --------------------------------------------------------
# AdvEnt training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import argparse
import os
import os.path as osp
import pprint
import warnings
import numpy as np
import mindspore.dataset as ds
from model.deeplabv2 import get_deeplab_v2
from dataset.cityscapes import CityscapesDataSet
from domain_adaptation.config import cfg, cfg_from_file
from domain_adaptation.eval_UDA import evaluate_domain_adaptation
from mindspore import context
from model_utils.device_adapter import get_device_id
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from model.train_model import WithEvalCellSrc

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    parser.add_argument("--device_target", type=str, default='Ascend')

    return parser.parse_args()


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


def main(config_file, exp_suffix):
    # LOAD ARGS
    assert config_file is not None, 'Missing cfg file'
    cfg_from_file(config_file)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'
    if exp_suffix:
        cfg.EXP_NAME += f'_{exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TEST.SNAPSHOT_DIR[0] == '':
        cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TEST.SNAPSHOT_DIR[0], exist_ok=True)

    if args.device_target == "CPU":
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    elif args.device_target == 'GPU':
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                            device_target="GPU", device_id=get_device_id())
    else:
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                            device_target="Ascend", device_id=get_device_id())

    print('Using config:')
    pprint.pprint(cfg)
    # load models
    models = []
    n_models = len(cfg.TEST.MODEL)
    if cfg.TEST.MODE == 'best':
        assert n_models == 1, 'Not yet supported'
    for i in range(n_models):
        if cfg.TEST.MODEL[i] == 'DeepLabv2':
            model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES,
                                   multi_level=cfg.TEST.MULTI_LEVEL[i])
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[i]}")
        models.append(model)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # dataloaders
    test_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                     list_path=cfg.DATA_LIST_TARGET,
                                     set=cfg.TEST.SET_TARGET,
                                     info_path=cfg.TEST.INFO_TARGET,
                                     crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                     mean=cfg.TEST.IMG_MEAN,
                                     labels_size=cfg.TEST.OUTPUT_SIZE_TARGET)

    test_loader = ds.GeneratorDataset(test_dataset, ["data", "label"], shuffle=False)
    test_loader = test_loader.batch(cfg.TEST.BATCH_SIZE_TARGET)

    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    # 60
    restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_iter060000_advent.ckpt')
    
    # restore_from = './pretrained/DeepLab_resnet_pretrained_imagenet_new.ckpt'
    
    print("Evaluating model", restore_from)
    model = models[0]
    
    # saved_state_dict = load_checkpoint(restore_from)
    # split_list = ['net_G', 'net_D1', 'net_D2']
    # train_state_dict = split_checkpoint(saved_state_dict, split_list=split_list)
    # load_param_into_net(model, train_state_dict['net_G'])
    # print('success load model !')
    
    saved_state_dict = load_checkpoint(restore_from)
    load_param_into_net(model, saved_state_dict)
    
    model.set_train(False)
    # eval
    eval_net = WithEvalCellSrc(model)
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
        
    index = 0
    for data in test_loader.create_dict_iterator():
        index = index + 1
        image, label = data['data'], data['label']
        # if not fixed_test_size:
        interp_size = (label.shape[1], label.shape[2])
        # else:
            # interp_size = (cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0])
        pred_main = eval_net(image, interp_size)
        pred = pred_main.argmax(axis=1)
        hist += cal_hist(data["label"].asnumpy().astype(np.int32).flatten(), pred.asnumpy().astype(np.int32).flatten(), cfg.NUM_CLASSES)
        print('processed {} images'.format(i + 1))
        i = i + 1
        
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mIoU = np.nanmean(iu)
    print('Val result: mIoU {:.4f}.'.format(mIoU))
    for i in range(cfg.NUM_CLASSES):
        print('Class_{} Result: iou {:.4f}.'.format(i, iu[i]))

    


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg, args.exp_suffix)
