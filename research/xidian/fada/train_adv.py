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

from core.models.fada_mindspore_bf import FADA_MindSpore_Feature_Extractor, FADA_MindSpore_Classifier, FADA_MindSpore_Discriminator
from core.models.fada_mindspore_bf import WithLossCellG, WithLossCellD, WithEvalCellSrc
from core.models.fada_mindspore_bf import CustomTrainOneStepCellG, CustomTrainOneStepCellD

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_adversarial_discriminator
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, cal_hist
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


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

    lr_iter_f = learning_rates.poly_lr(cfg.SOLVER.BASE_LR, cfg.SOLVER.MAX_ITER, cfg.SOLVER.MAX_ITER, end_lr=0.0,
                                       power=cfg.SOLVER.LR_POWER)

    lr_iter_c = learning_rates.poly_lr(cfg.SOLVER.BASE_LR * 10, cfg.SOLVER.MAX_ITER, cfg.SOLVER.MAX_ITER, end_lr=0.0,
                                       power=cfg.SOLVER.LR_POWER)

    lr_iter_d = learning_rates.poly_lr(cfg.SOLVER.BASE_LR_D, cfg.SOLVER.MAX_ITER, cfg.SOLVER.MAX_ITER, end_lr=0.0,
                                       power=cfg.SOLVER.LR_POWER)

    opt_f = nn.SGD(feature_extractor.trainable_params(), learning_rate=lr_iter_f, momentum=cfg.SOLVER.MOMENTUM,
                   weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    opt_c = nn.SGD(classifier.trainable_params(), learning_rate=lr_iter_c, momentum=cfg.SOLVER.MOMENTUM,
                   weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    opt_d = nn.Adam(discriminator.trainable_params(), learning_rate=lr_iter_d, beta1=0.9, beta2=0.99)

    criterion = LOss.SoftmaxCrossEntropyLoss(num_cls=cfg.MODEL.NUM_CLASSES, ignore_label=255)

    netG_with_loss = WithLossCellG(feature_extractor,
                                   classifier,
                                   discriminator,
                                   criterion)

    netD_with_loss = WithLossCellD(feature_extractor,
                                   classifier,
                                   discriminator)
    
    netG_with_loss.set_grad()
    netD_with_loss.set_grad()
    
    train_netG = CustomTrainOneStepCellG(netG_with_loss, opt_f, opt_c)

    train_netD = CustomTrainOneStepCellD(netD_with_loss, opt_d)
    
    train_netG.set_train()
    train_netD.set_train()
    
    max_iters = cfg.SOLVER.MAX_ITER
    output_dir = cfg.OUTPUT_DIR
    start_epoch = 0
    iteration = 0

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")

    start_training_time = time.time()
    end = time.time()
    for i, datas in enumerate(zip(data_set_src.create_dict_iterator(), data_set_trg.create_dict_iterator())):
        data_src, data_trg = datas
        data_time = time.time() - end

        src_input = data_src['data']
        src_label = data_src['label']
        tgt_input = data_trg['data']
        src_size = src_input.shape[-2:]
        tgt_size = tgt_input.shape[-2:]
        
        loss_seg, loss_adv_tgt, src_fea, tgt_fea, src_soft_label, tgt_soft_label = train_netG(src_input, tgt_input, src_label, src_size, tgt_size)
        loss_D_src, loss_D_tgt = train_netD(src_fea, tgt_fea, src_soft_label, tgt_soft_label, src_size, tgt_size)
        
        # loss_D_src, loss_D_tgt = train_netD(src_input, tgt_input, src_size, tgt_size)
        # loss_seg, loss_adv_tgt = train_netG(src_input, tgt_input, src_label, src_size, tgt_size)

        iteration = iteration + 1

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(f"eta: {eta_string}, "
                        f"iter: [{iteration} / {max_iters}], "
                        f"loss_seg: {loss_seg}, "
                        f"loss_adv_tgt: {loss_adv_tgt}, "
                        f"loss_D: {loss_D_src+loss_D_tgt}, "
                        f"loss_D_src: {loss_D_src}, "
                        f"loss_D_tgt: {loss_D_tgt}, ")
                
        if iteration == cfg.SOLVER.MAX_ITER or iteration % cfg.SOLVER.CHECKPOINT_PERIOD==0:
            filename_f = os.path.join(output_dir, "model_iter{:06d}_feature_extractor.ckpt".format(iteration))
            filename_c = os.path.join(output_dir, "model_iter{:06d}_classifier.ckpt".format(iteration))
            filename_d = os.path.join(output_dir, "model_iter{:06d}_discriminator.ckpt".format(iteration))
            mindspore.save_checkpoint(feature_extractor, filename_f)
            mindspore.save_checkpoint(classifier, filename_c)
            mindspore.save_checkpoint(discriminator, filename_d)
        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break
            
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (cfg.SOLVER.MAX_ITER)
        )
    )

    return feature_extractor, classifier          


def run_test(cfg, feature_extractor, classifier):
    logger = logging.getLogger("FADA.tester")
    logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')

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

    # 真正验证迭代过程
    i = 0
    for data in data_set.create_dict_iterator():

        pred = eval_net(data["data"], data["label"])
        pred = pred.argmax(axis=1)

        hist += cal_hist(data["label"].asnumpy().astype(np.int32).flatten(), pred.asnumpy().astype(np.int32).flatten(), cfg.MODEL.NUM_CLASSES)
        logger.info('processed {} images'.format(i+1))
        i = i + 1

    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mIoU = np.nanmean(iu)
    logger.info('Val result: mIoU {:.4f}.'.format(mIoU))
    for i in range(cfg.MODEL.NUM_CLASSES):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i, iu[i]))


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

    fea, clse = train(cfg)

    if not args.skip_test:
        run_test(cfg, fea, clse)


if __name__ == "__main__":
    main()
