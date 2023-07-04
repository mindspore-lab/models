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

import argparse
import os
import os.path as osp
import pprint
import warnings

import yaml
import mindspore

from model.deeplabv2 import get_deeplab_v2
from dataset.gta5 import GTA5DataSet
from dataset.cityscapes import CityscapesDataSet
from domain_adaptation.config import cfg, cfg_from_file
from domain_adaptation.train_UDA import train_domain_adaptation

from mindspore import context
import mindspore.dataset as ds
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from model_utils.device_adapter import get_device_id


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


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


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    parser.add_argument("--device_target", type=str, default='Ascend')

    return parser.parse_args()


def main():
    # LOAD ARGS
    args = get_arguments()
    print('Called with args:')
    print(args)

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
        print('Using config:')
    pprint.pprint(cfg)

    # INIT
    _init_fn = None

    # LOAD SEGMENTATION NET
    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'
    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = load_checkpoint(cfg.TRAIN.RESTORE_FROM)
        split_list = ['net_G', 'net_D1', 'net_D2']
        train_state_dict = split_checkpoint(saved_state_dict, split_list=split_list)
        load_param_into_net(model, train_state_dict['net_G'])
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    print('Model loaded')

    if args.device_target == "CPU":
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    elif args.device_target == 'GPU':
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                            device_target="GPU", device_id=get_device_id())
    else:
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                            device_target="Ascend", device_id=get_device_id())

    # DATALOADERS
    source_dataset = GTA5DataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                 list_path=cfg.DATA_LIST_SOURCE,
                                 set=cfg.TRAIN.SET_SOURCE,
                                 max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                 crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                 mean=cfg.TRAIN.IMG_MEAN)

    source_loader = ds.GeneratorDataset(source_dataset, ["data", "label"], shuffle=True)
    source_loader = source_loader.batch(cfg.TRAIN.BATCH_SIZE_SOURCE)


    target_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set = cfg.TRAIN.SET_TARGET,
                                       info_path=cfg.TRAIN.INFO_TARGET,
                                       max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
                                       crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN)

    target_loader = ds.GeneratorDataset(target_dataset, ["data", "label"], shuffle=True)
    target_loader = target_loader.batch(cfg.TRAIN.BATCH_SIZE_TARGET)

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    conv_params_fea = list(filter(lambda x: ('layer1' in x.name) or ('layer2' in x.name)
                                            or ('layer3' in x.name) or ('layer4' in x.name),
                                  model.trainable_params()))

    conv_params_cls = list(filter(lambda x: ('layer5' in x.name) or ('layer6' in x.name), model.trainable_params()))
    
    print(1)
    
    # UDA TRAINING
    train_domain_adaptation(model, source_loader, target_loader, cfg)


if __name__ == '__main__':
    main()
