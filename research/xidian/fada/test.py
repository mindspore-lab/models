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
from tqdm import tqdm
from collections import OrderedDict
import re

import mindspore
from mindspore import context

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_model, build_adversarial_discriminator, build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, cal_hist, get_color_pallete
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger

from core.models.fada_mindspore import FADA_MindSpore_Feature_Extractor, FADA_MindSpore_Classifier, \
    FADA_MindSpore_Discriminator
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from core.models.fada_mindspore import WithLossCellG, WithLossCellD, WithEvalCellSrc
import mindspore.dataset as ds
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


def multi_scale_inference(feature_extractor, classifier, image, label, scales=[0.7, 1.0, 1.3], flip=True):
    output = None
    size = image.shape[-2:]
    for s in scales:
        x = F.interpolate(image, size=(int(size[0] * s), int(size[1] * s)), mode='bilinear', align_corners=True)
        pred = inference(feature_extractor, classifier, x, label, flip=False)
        if output is None:
            output = pred
        else:
            output = output + pred
        if flip:
            x_flip = torch.flip(x, [3])
            pred = inference(feature_extractor, classifier, x_flip, label, flip=False)
            output = output + pred.flip(3)
    if flip:
        return output / len(scales) / 2
    return output / len(scales)


def transform_color(pred):
    synthia_to_city = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 15,
        14: 17,
        15: 18,
    }
    label_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in synthia_to_city.items():
        label_copy[pred == k] = v
    return label_copy.copy()


def test(cfg, saveres):
    logger = logging.getLogger("FADA.tester")
    logger.info("Start testing")

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

    dataset_name = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)
    test_data = build_dataset(cfg, mode='test', is_source=False)
    data_set = ds.GeneratorDataset(test_data, ["data", "label"], shuffle=False)
    data_set = data_set.batch(cfg.TEST.BATCH_SIZE)

    eval_net = WithEvalCellSrc(feature_extractor, classifier)
    eval_net.set_train(False)

    hist = np.zeros((cfg.MODEL.NUM_CLASSES, cfg.MODEL.NUM_CLASSES))
    data_list = './datasets/cityscape/cityscapes_train_list.txt'
    with open(data_list, "r") as handle:
        content = handle.readlines()

    i = 0
    for data in data_set.create_dict_iterator():

        # pred = inference(eval_net, data["data"], data["label"])
    
        pred = eval_net(data["data"], data["label"])
        # print(pred.shape)
        pred = pred.argmax(axis=1)

        hist += cal_hist(data["label"].asnumpy().astype(np.int32).flatten(), pred.asnumpy().astype(np.int32).flatten(),
                         cfg.MODEL.NUM_CLASSES)
        logger.info('processed {} images'.format(i + 1))
        name = content[i].strip()

        i = i + 1
        if saveres:
            pred = pred.asnumpy().squeeze()
            # uncomment the following line when visualizing SYNTHIA->Cityscapes
            # pred = transform_color(pred)
            mask = get_color_pallete(pred, "city")
            mask_filename = name if len(name.split("/")) < 2 else name.split("/")[1]
            mask.save(os.path.join(output_folder, mask_filename))

    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mIoU = np.nanmean(iu)
    logger.info('Val result: mIoU {:.4f}.'.format(mIoU))
    for i in range(cfg.MODEL.NUM_CLASSES):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i, iu[i]))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument('--saveres', action="store_true",
                        help='save the result')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("FADA", save_dir, 0)
    logger.info(cfg)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                        device_target="Ascend")

    test(cfg, args.saveres)


if __name__ == "__main__":
    main()

