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
import datetime
import logging
import time
import math
import numpy as np
from collections import OrderedDict

import mindspore
from mindspore import nn
from mindspore import context
from model_utils.device_adapter import get_device_id
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import mindspore.dataset as ds
from src.loss import loss as LOss
from src.utils import learning_rates

from core.models.fada_mindspore import FADA_MindSpore_Feature_Extractor, FADA_MindSpore_Classifier, FADA_MindSpore_Discriminator
from core.models.fada_mindspore import WithLossCellG, WithLossCellD, WithEvalCellSrc
from core.models.fada_mindspore import CustomTrainOneStepCellG, CustomTrainOneStepCellD

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_adversarial_discriminator
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, cal_hist
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from mindspore import ops
from PIL import Image

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def inference(eval_net, image, label, flip=True):

    image = ops.Concat(axis=0)((image, ops.ReverseV2(axis=[3])(image)))
    output = eval_net(image, label)

    output = ops.Softmax(axis=1)(output)
    output = (output[0] + ops.ReverseV2(axis=[2])(output[1])) / 2
    return output


def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))


def train(cfg):
    logger = logging.getLogger("FADA.trainer")
    logger.info("Start training")

    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)

    batch_size = cfg.SOLVER.BATCH_SIZE // 2

    data_set_src = ds.GeneratorDataset(src_train_data, ["data", "label"], shuffle=True)
    data_set_src = data_set_src.batch(batch_size)

    data_set_trg = ds.GeneratorDataset(tgt_train_data, ["data", "label"], shuffle=True)
    data_set_trg = data_set_trg.batch(batch_size)

    feature_extractor = FADA_MindSpore_Feature_Extractor(cfg)

    classifier = FADA_MindSpore_Classifier(cfg)

    discriminator = FADA_MindSpore_Discriminator(cfg)


    if cfg.resume_f:
        logger.info("Loading checkpoint from {}".format(cfg.resume_f))
        checkpoint = load_checkpoint(cfg.resume_f)
        load_param_into_net(feature_extractor, checkpoint)
    if cfg.resume_c:
        logger.info("Loading checkpoint from {}".format(cfg.resume_c))
        checkpoint = load_checkpoint(cfg.resume_c)
        load_param_into_net(classifier, checkpoint)
    if cfg.resume_d:
        logger.info("Loading checkpoint from {}".format(cfg.resume_f))
        checkpoint = load_checkpoint(cfg.resume_d)
        load_param_into_net(discriminator, checkpoint)

    
    max_iters = cfg.SOLVER.MAX_ITER
    output_dir = cfg.OUTPUT_DIR
    start_epoch = 0
    iteration = 0

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")

    start_training_time = time.time()
    end = time.time()
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (cfg.SOLVER.MAX_ITER)
        )
    )

    return feature_extractor, classifier          


def run_test(cfg):
    logger = logging.getLogger("FADA.tester")
    logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')
    
    feature_extractor = FADA_MindSpore_Feature_Extractor(cfg)

    classifier = FADA_MindSpore_Classifier(cfg)

    discriminator = FADA_MindSpore_Discriminator(cfg)
    miou = 0
    pos = 0
    
    for ii in range(264, 265, 2):
        f_ckpt = './results/adv_test_d19/model_iter0%d00_feature_extractor.ckpt' % ii
        c_ckpt = './results/adv_test_d19/model_iter0%d00_classifier.ckpt' % ii
        # f_ckpt = './results/sd_test_new_advd19/model_iter0%d00_feature_extractor.ckpt' % ii
        # c_ckpt = './results/sd_test_new_advd19/model_iter0%d00_classifier.ckpt' % ii
        logger.info("Loading checkpoint from {}".format(f_ckpt))
        checkpoint = load_checkpoint(f_ckpt)
        load_param_into_net(feature_extractor, checkpoint)
        logger.info("Loading checkpoint from {}".format(c_ckpt))
        checkpoint = load_checkpoint(c_ckpt)
        load_param_into_net(classifier, checkpoint)
    
    
        dataset_name = cfg.DATASETS.TEST
        
        test_data = build_dataset(cfg, mode='test', is_source=False)
        data_set = ds.GeneratorDataset(test_data, ["data", "label"], shuffle=False)
        data_set = data_set.batch(cfg.TEST.BATCH_SIZE)

        eval_net = WithEvalCellSrc(feature_extractor, classifier)
        eval_net.set_train(False)

        hist = np.zeros((cfg.MODEL.NUM_CLASSES, cfg.MODEL.NUM_CLASSES))

        # 真正验证迭代过程
        i = 0
        for data in data_set.create_dict_iterator():

            # pred = eval_net(data["data"], data["label"])
            # pred = pred.argmax(axis=1)
            
            pred = inference(eval_net, data["data"], data["label"])
    
            # pred = eval_net(data["data"], data["label"])
            # print(pred.shape)
            pred = pred.argmax(axis=0)
            
            hist += cal_hist(data["label"].asnumpy().astype(np.int32).flatten(), pred.asnumpy().astype(np.int32).flatten(), cfg.MODEL.NUM_CLASSES)
            logger.info('processed {} images'.format(i+1))
            i = i + 1

        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mIoU = np.nanmean(iu)
        logger.info('Val result: mIoU {:.4f}.'.format(mIoU))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info('Class_{} Result: iou {:.4f}.'.format(i, iu[i]))
            
        if mIoU > miou:
            miou = mIoU
            pos = ii
        
        print("mIoU:", miou)
        print('pos', pos)
        
    print("mIoU:", miou)
    print('pos', pos)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--device_target", type=str, default='CPU')

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("FADA", output_dir, args.local_rank)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if args.device_target == "CPU":
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    elif args.device_target == 'GPU':
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                            device_target="GPU", device_id=get_device_id())
    else:
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                            device_target="Ascend", device_id=get_device_id())

    
    run_test(cfg)


if __name__ == "__main__":
    main()
