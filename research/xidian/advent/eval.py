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

import mindspore.dataset as ds
from src.nets.deeplabv2 import get_deeplab_v2
from src.datasets.cityscapes import CityscapesDataSet
from src.domain_adaptation.config import cfg, cfg_from_file
from src.domain_adaptation.eval_UDA import evaluate_domain_adaptation
from mindspore import context,load_param_into_net,load_checkpoint



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
    parser.add_argument("--device_id", type=int, default=6)
    parser.add_argument("--restore_from",type=str,default='')

    return parser.parse_args()


def main(config_file, exp_suffix):
    # LOAD ARGS
    # assert config_file is not None, 'Missing cfg file'
    # cfg_from_file(config_file)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'
    if exp_suffix:
        cfg.EXP_NAME += f'_{exp_suffix}'

    cfg.TEST.RESTORE_FROM = args.restore_from if args.restore_from else cfg.TEST.RESTORE_FROM
    # print(cfg.TEST.RESTORE_FROM)
    # raise None
    if args.device_target == "CPU":
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    elif args.device_target == 'GPU':
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                            device_target="GPU", device_id=args.device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                            device_target="Ascend", device_id=args.device_id)

    print('Using config:')
    pprint.pprint(cfg)
    # load models

    if cfg.TEST.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES,
                               multi_level=cfg.TEST.MULTI_LEVEL)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL}")

    # load model
    print('restore model from {}'.format(cfg.TEST.RESTORE_FROM))
    saved_state_dict = load_checkpoint(cfg.TEST.RESTORE_FROM)
    load_param_into_net(model, saved_state_dict)
    print('load model success')

    # dataloaders
    test_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                     list_path=cfg.DATA_LIST_TARGET,
                                     set=cfg.TEST.SET_TARGET,
                                     info_path=cfg.TEST.INFO_TARGET,
                                     crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                     mean=cfg.TEST.IMG_MEAN,
                                     labels_size=cfg.TEST.OUTPUT_SIZE_TARGET)

    test_loader = ds.GeneratorDataset(test_dataset, ["data", "label","name"], shuffle=False)
    test_loader = test_loader.batch(cfg.TEST.BATCH_SIZE_TARGET)

    # eval
    evaluate_domain_adaptation(model, test_loader, cfg)


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg, args.exp_suffix)
