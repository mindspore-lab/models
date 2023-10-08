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

import numpy as np
import mindspore
import mindspore.dataset as ds
import mindspore.ops as ops
from options.evaluate_options import TestOptions
from dataset.cityscapes_dataset import cityscapesDataSet
import os
from PIL import Image
from os.path import join
import json



IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def evaluation(model, testloader, interp, data_dir, save_path, devkit_dir, logger=None, save=False):
    model.set_train(False)
    os.makedirs(save_path, exist_ok=True)
    preds_list = list()

    image_path_list = join(devkit_dir, 'val.txt')
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [x.split('/')[-1] for x in pred_imgs]

    for index, batch in enumerate(testloader):
        # if index>=5:
        #     break
        if index % 100 == 0:
            print('%d processd' % index)
            if logger:
                logger.write('%d processd\n' % index)
        image = batch['image']
        name = pred_imgs[index]
        output1, output2 = model(image)
        output = interp(output2).asnumpy()
        n, c, h, w = output.shape
        output = output.reshape((c, h, w))
        output = output.transpose((1, 2, 0))

        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        if save:
            output_col = colorize_mask(output)
            output_col.save('%s/%s_color.png' % (save_path, name.split('.')[0]))
        output = Image.fromarray(output)
        preds_list.append(output)

    gt_path = os.path.join(data_dir, 'gtFine', 'val')
    # print(preds_dict.keys())
    mIoUs, log = compute_mIoU(gt_dir=gt_path, preds=preds_list, devkit_dir=devkit_dir)
    if logger:
        logger.write(log + '\n')
    model.set_train(True)
    return round(np.nanmean(mIoUs) * 100, 2)


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
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]

    image_path_list = join(devkit_dir, 'val.txt')
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [x.split('/')[-1] for x in pred_imgs]

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
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)
    log = ''
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        log = log + '\n' + str('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    log = log + '\n' + '===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))
    return mIoUs, log


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


def main():
    """Create the model and start the evaluation process."""

    opt = TestOptions()
    args = opt.initialize()

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)
    w, h = map(int, args.output_size.split(','))
    output_size = (w, h)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # model = DeeplabMulti(num_classes=args.num_classes)
    model = get_deeplab_v2(num_classes=args.num_classes)

    print('model path:',args.restore_from)
    if args.restore_from:
        saved_state_dict = mindspore.load_checkpoint(args.restore_from)
        split_list = ['net_G', 'net_D1', 'net_D2']
        train_state_dict = split_checkpoint(saved_state_dict, split_list=split_list)
        mindspore.load_param_into_net(model, train_state_dict['net_G'])
        print('success load model !')

    # print(model)
    cityscapes_generator = cityscapesDataSet(args.data_dir, args.data_list,
                                             crop_size=input_size, scale=False,
                                             mirror=False, mean=IMG_MEAN,
                                             set=args.set)
    cityscapes_dataset = ds.GeneratorDataset(cityscapes_generator, shuffle=True,
                                             column_names=['image', 'size'])
    cityscapes_dataset = cityscapes_dataset.batch(batch_size=1)
    target_iterator = cityscapes_dataset.create_dict_iterator()
    interp = ops.ResizeBilinear(size=output_size)
    model.set_train(False)
    evaluation(model=model,
               testloader=target_iterator,
               interp=interp,
               data_dir=args.data_dir,
               save_path=args.save_path,
               devkit_dir=args.devkit_dir, save=False)


# evaluation(model, testloader, interp, data_dir, save_path, devkit_dir, logger=None, save=True):
if __name__ == '__main__':
    main()
