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

import os.path as osp
import time
import mindspore.nn as nn

import numpy as np
from tqdm import tqdm

from utils.func import per_class_iu, fast_hist
from utils.serialization import pickle_dump, pickle_load
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from model.train_model import WithEvalCellSrc


def evaluate_domain_adaptation( models, test_loader, cfg,
                                fixed_test_size=True,
                                verbose=True):
    interp = None

    if fixed_test_size:
        interp = nn.ResizeBilinear()
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    test_loader, interp, fixed_test_size,
                    verbose)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  test_loader, interp, fixed_test_size,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_single(cfg, models,
                device, test_loader, interp,
                fixed_test_size, verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for index, batch in tqdm(enumerate(test_loader)):
        image, label, _, name = batch
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                pred_main = model(image.cuda(device))[1]
                output_ = interp(pred_main).cpu().data[0].numpy()
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
    inters_over_union_classes = per_class_iu(hist)
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
    if verbose:
        display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)


def eval_best(cfg, models,
              test_loader, interp,
              fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res.pkl')
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    cur_best_miou = -1
    cur_best_model = ''
    for i_iter in range(5000, 120000, step):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_iter{i_iter:06d}_advent.ckpt')
        # restore_from = './pretrained/DeepLab_resnet_pretrained_imagenet_new.ckpt'

        if not osp.exists(restore_from):
            print(restore_from)
            continue
            # if cfg.TEST.WAIT_MODEL:
            #     print('Waiting for model..!')
            #     while not osp.exists(restore_from):
            #         time.sleep(5)
        print("Evaluating model", restore_from)
        if i_iter not in all_res.keys():
            model = models[0]
            saved_state_dict = load_checkpoint(restore_from)
            load_param_into_net(model, saved_state_dict)
            model.set_train(False)
            # eval
            eval_net = WithEvalCellSrc(model)
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            # for index, batch in enumerate(test_loader):
            #     image, _, _, name = batch
            index = 0
            for data in test_loader.create_dict_iterator():
                index = index + 1
                image, label = data['data'], data['label']
                if not fixed_test_size:
                    interp_size = (label.shape[1], label.shape[2])
                else:
                    interp_size = (cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0])
                pred_main = eval_net(image, interp_size)
                output = pred_main[0].asnumpy()
                output = output.transpose(1, 2, 0)
                output = np.argmax(output, axis=2)
                label = label.asnumpy()[0]
                hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                if verbose and index > 0 and index % 100 == 0:
                    print('{:d} : {:0.2f}'.format(
                        index, 100 * np.nanmean(per_class_iu(hist))))
            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = inters_over_union_classes
            pickle_dump(all_res, cache_path)
        else:
            inters_over_union_classes = all_res[i_iter]
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from
        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)
        


def load_checkpoint_for_evaluation(model, checkpoint):
    saved_state_dict = load_checkpoint(checkpoint)
    load_param_into_net(model, saved_state_dict)
    model.set_train(False)



def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
