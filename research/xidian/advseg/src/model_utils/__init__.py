# Copyright 2020 Huawei Technologies Co., Ltd
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

import numpy as np
import json
from PIL import Image
import os
import time
from os.path import join
from src.utils.softmax import Softmax
from .local_adapter import get_device_id, get_rank_id, get_job_id, get_device_num
from .config import config


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


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


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, preds, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    # print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]

    image_path_list = join(devkit_dir, 'val.txt')
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [x.split('/')[-1] for x in pred_imgs]

    if len(preds) != len(gt_imgs):
        print('The number not equal: preds={} and gt_imgs={}'.format(len(preds), len(gt_imgs)))
        raise ValueError('The number not equal: preds={} and gt_imgs={}'.format(len(preds), len(gt_imgs)))
    for ind in range(len(gt_imgs)):
        pred = np.asarray(preds[ind])
        label = np.asarray(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # if ind > 0 and ind % 10 == 0:
        #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)
    log = ''
    # if (config.device_target == 'CPU') or get_rank() == 0:
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        log = log + '\n' + str('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    log = log + '\n' + '===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))
    return mIoUs, log


def evaluation(model, testloader, interp, data_dir_target, save_result, data_list_target, logger=None, save=False, config=None):
    # mindspore.context.set_context(dataset_strategy='full_batch')
    model.set_train(False)
    os.makedirs(save_result, exist_ok=True)
    preds_list = list()

    image_path_list = join(data_list_target, 'val.txt')
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [x.split('/')[-1] for x in pred_imgs]

    all_time = 0.
    all_index = 0
    for index, batch in enumerate(testloader):
        # if index>=5:
        #     break
        start_time = time.time()

        image = batch['image']
        name = pred_imgs[index]
        # name = batch['name']
        output1, output2 = model(image)
        output = interp(output2).asnumpy()
        n, c, h, w = output.shape
        output = output.reshape((c, h, w))
        output = output.transpose((1, 2, 0))

        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        if save:
            output_col = colorize_mask(output)
            output_col.save('%s/%s_color.png' % (save_result, name.split('.')[0]))
        output = Image.fromarray(output)
        preds_list.append(output)
        end_time = time.time()
        batch_time = end_time - start_time
        all_time += batch_time
        all_index += 1
        if (index % 10 == 0) and ((not config) or (not config.is_distributed) or (config.rank == 0)):
            print('batch_id: %d processed \t Time: %.6f' % (index, batch_time))
        # elif not config.is_distributed:
        #     print('batch_id: %d processed \t Time: %.6f' % (index, batch_time))
        # elif get_rank()==0:

        if (index % 10 == 0) and logger:
            logger.write('batch_id: %d processed \t Time: %.6f \n' % (index, batch_time))

    avg_time = all_time / all_index
    if (not config) or (not config.is_distributed) or (config.rank == 0):
        print('Total Image: {}\t Total processed time: {}\t Avg processed time: {}\n'.format(all_index, all_time, avg_time))

    # if not config:
    #     print('Total Image: {}\t Total processed time: {}\t Avg processed time: {}\n'.format(all_index, all_time, avg_time))
    # elif not config.is_distributed :
    #     print('Total Image: {}\t Total processed time: {}\t Avg processed time: {}\n'.format(all_index, all_time, avg_time))
    # elif get_rank()==0:
    #     print('Total Image: {}\t Total processed time: {}\t Avg processed time: {}\n'.format(all_index, all_time, avg_time))

    if os.path.exists(os.path.join(data_dir_target, 'gtFine_trainvaltest')):
        gt_path = os.path.join(data_dir_target, 'gtFine_trainvaltest', 'gtFine', 'val')
    else:
        gt_path = os.path.join(data_dir_target, 'gtFine', 'val')
    # print(preds_dict.keys())
    mIoUs, log = compute_mIoU(gt_dir=gt_path, preds=preds_list, devkit_dir=data_list_target)
    if logger:
        logger.write(log + '\n')
    model.set_train(True)
    # mindspore.context.set_context(dataset_strategy='data_parallel')
    return round(np.nanmean(mIoUs) * 100, 2)
