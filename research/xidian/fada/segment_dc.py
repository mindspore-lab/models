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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
from PIL import Image
import numpy as np
import skimage.io as io
import cv2 as cv
import sys
import random
import warnings
import argparse
warnings.filterwarnings("ignore")

def colorize_mask(mask):
    # mask: numpy array of the mask
    # classes num 6 : palette = [255,0,0, 255,255,255, 0, 0, 255, 0, 255, 0,255,255,0,0,255,255]
    # classes num 5 :
    palette = [
        0, 0, 0,
        0, 0, 255,  # building 建筑
        70, 70, 70,  # water 水体
        128, 64, 128,  # road 道路
        107, 142, 35,  # vegetation 植被
        250, 170, 30,  # plough 耕地
        0, 200, 0,  # "lawn" 草地
        0, 0, 142,  # "soil" 裸土
        0, 0, 100,  # airport 机场
        0, 255, 255,  # other 其他用地
        0, 0, 230  # playground 操场
    ]
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = np.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


Image.MAX_IMAGE_PIXELS = 100000000000

parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_name', required=True, type=str)
parser.add_argument('--train_val', required=True, type=str)

args = parser.parse_args()


dataset_name = args.dataset_name

# train_val = 'train'
train_val = args.train_val

save_base_path = '/media/amax/wzn/datasets/tb/{}/'.format(dataset_name)
if not os.path.exists(save_base_path):
    os.mkdir(save_base_path)


base_images = save_base_path + '{}_images/'.format(train_val)
if not os.path.exists(base_images):
    os.mkdir(base_images)

base_gt = save_base_path + '{}_gt/'.format(train_val)
if not os.path.exists(base_gt):
    os.mkdir(base_gt)

base_gt_vis = save_base_path + 'gt_vis_{}/'.format(train_val)
if not os.path.exists(base_gt_vis):
    os.mkdir(base_gt_vis)

ori_images = './{}/{}/'.format(dataset_name, train_val)
ori_gt = './{}/{}/'.format(dataset_name, train_val)
total_num = 0
img_len_idx = [0]
imglist = os.listdir(ori_images)

color_map = [
    0, 0, 0,
    0, 0, 255,  # building
    70, 70, 70,  # water
    128, 64, 128,  # road
    107, 142, 35,  # vegetation
    250, 170, 30,  # plough
    0, 200, 0,  # "lawn"
    0, 0, 142,  # "soil"
    0, 0, 100,  # airport
    0, 255, 255,  # other
    0, 0, 230  # playground 31->10
]

class_name = ["background", \
              "building", \
              "water", \
              "road", \
              "vegetation",
              "plough",
              "lawn",
              "soil",
              "Airport",
              "other land",
              "playground"]

for train_img in imglist:

    if train_img[-4:] == '.tif' and 'fda' not in train_img and 'label' not in train_img:
        print('start segment image_{} .....'.format(train_img))
        img_ori = io.imread(ori_images + train_img)
        print(img_ori.shape)
        print(img_ori.dtype)

        img_ori = img_ori[:, :, :3]
        # min max norm
        img_ori = (img_ori.astype(np.float) - np.min(img_ori)) / (np.max(img_ori) - np.min(img_ori))
        img_ori = (img_ori * 255).astype(np.uint8)
        # liner
        # smax,smin = np.max(img_ori), np.min(img_ori)
        # dmax,dmin = 255, 0
        # img_ori = img_ori.astype(np.float) * (dmax-dmin)/(smax-smin) + (smax*dmin - smin*dmax)/(smax-smin)
        # img_ori = img_ori.astype(np.uint8)

        print(np.max(img_ori))
        # break

        # read label
        label_name = train_img.replace(".tif", '_label.tif')
        anno_map_ori = Image.open(ori_gt + label_name)  # 注意修改label路径
        anno_map = np.asarray(anno_map_ori).copy()
        anno_map[anno_map == 31] = 10
        print(anno_map.shape)


        unit_size = 256  # 窗口大小
        window_step = 256

        scales = [1]
        for scale in scales:
            anno_map = cv2.resize(anno_map, (img_ori.shape[1], img_ori.shape[0]), interpolation=cv2.INTER_NEAREST)
            anno_map[np.sum(img_ori == 0) == 3] = 0
            w, h = anno_map.shape[:2]

            img = cv2.resize(img_ori, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            # anno_map_vis = colorize_mask(anno_map)
            # anno_map_vis.save('cq_B_vis.tif'.format(total_num))
            # exit()

            length, width = img.shape[0], img.shape[1]
            assert length == w and width == h
            # 滑动窗口
            x1, x2, y1, y2 = 0, unit_size, 0, unit_size
            count = 0
            while (x1 < length):
                # 判断横向是否越界
                if x2 > length:
                    x2, x1 = length, length - unit_size

                while (y1 < width):
                    if y2 > width:
                        y2, y1 = width, width - unit_size

                    if np.average(anno_map[x1:x2, y1:y2] == 0) < 0.9:
                        io.imsave(base_images + '{:03d}.tif'.format(total_num), img[x1:x2, y1:y2])
                        anno_map_vis = colorize_mask(anno_map[x1:x2, y1:y2])
                        anno_map_vis.save(base_gt_vis + '{:03d}.tif'.format(total_num))
                        cv.imwrite(base_gt + '{:03d}.tif'.format(total_num), anno_map[x1:x2, y1:y2])
                        count += 1
                        total_num += 1

                    if y2 == width: break

                    y1 += window_step
                    y2 += window_step

                if x2 == length: break

                y1, y2 = 0, unit_size
                x1 += window_step
                x2 += window_step
        #            view_bar('gen image_{} {}*{} pic---'.format(train_img_num,unit_size,unit_size), x2//window_step, length//window_step)
        #            print('\n')

        img_len_idx.append(total_num)

print('gen list ....... pic {}'.format(img_len_idx))
