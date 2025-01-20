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

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_model, build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, cal_hist
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger

import mindspore
from mindspore import nn
from mindspore import context
from model_utils.device_adapter import get_device_id
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import mindspore.dataset as ds
from src.loss import loss as LOss
from src.utils import learning_rates

from core.models.fada_mindspore import FADA_MindSpore_Feature_Extractor, FADA_MindSpore_Classifier
from core.models.fada_mindspore import WithLossCellSrc, WithEvalCellSrc
from core.models.fada_mindspore import CustomTrainOneStepCellSrc


def train(cfg):
    
    logger = logging.getLogger("FADA.trainer")

    
    feature_extractor = FADA_MindSpore_Feature_Extractor(cfg)

    classifier = FADA_MindSpore_Classifier(cfg)

    if cfg.resume_f:
        logger.info("Loading checkpoint from {}".format(cfg.resume_f))
        checkpoint = load_checkpoint(cfg.resume_f)
        load_param_into_net(feature_extractor, checkpoint)
    if cfg.resume_c:
        logger.info("Loading checkpoint from {}".format(cfg.resume_c))
        checkpoint = load_checkpoint(cfg.resume_c)
        load_param_into_net(classifier, checkpoint)

   
    return feature_extractor, classifier


def run_test(cfg, feature_extractor, classifier):
    logger = logging.getLogger("FADA.tester")
    logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')

    dataset_name = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)
    test_data = build_dataset(cfg, mode='test', is_source=False)
    data_set = ds.GeneratorDataset(test_data, ["data", "label"], shuffle=True)
    data_set = data_set.batch(cfg.TEST.BATCH_SIZE)

    eval_net = WithEvalCellSrc(feature_extractor, classifier)
    eval_net.set_train(False)

    hist = np.zeros((cfg.MODEL.NUM_CLASSES, cfg.MODEL.NUM_CLASSES))

    # 真正验证迭代过程
    i = 0
    for data in data_set.create_dict_iterator():
        pred = eval_net(data["data"], data["label"])
        
        pred = pred.argmax(axis=1)
        hist += cal_hist(data["label"].asnumpy().astype(np.int32).flatten(), pred.asnumpy().astype(np.int32).flatten(),
                         cfg.MODEL.NUM_CLASSES)
        logger.info('processed {} images'.format(i + 1))
        i = i + 1
        
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mIoU = np.nanmean(iu)
    logger.info('Val result: mIoU {:.4f}.'.format(mIoU))
    for i in range(cfg.MODEL.NUM_CLASSES):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i, iu[i]))


def main():
    parser = argparse.ArgumentParser(description="Mindspore Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--device_target", type=str, default='CPU')

    args = parser.parse_args()

    num_gpus = 1
    args.distributed = False

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("FADA", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
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

    feature_extractor, classifier = train(cfg)


    if not args.skip_test:
        run_test(cfg, feature_extractor, classifier)


if __name__ == "__main__":
    main()
