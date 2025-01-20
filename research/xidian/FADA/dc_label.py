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

import os
import cv2
from PIL import Image
import numpy as np
import skimage.io as io
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_name', required=True, type=str)

args = parser.parse_args()

dataset_name = args.dataset_name

A_images = './{}/A_gt/'.format(dataset_name)
B_images = './{}/B_gt/'.format(dataset_name)

imglist = os.listdir(A_images)

save_base_path = './{}/labels/'.format(dataset_name)
save_vis_path = './{}/vis_labels/'.format(dataset_name)

if not os.path.exists(save_base_path):
    os.mkdir(save_base_path)
if not os.path.exists(save_vis_path):
    os.mkdir(save_vis_path)

for train_img in imglist:

    img_A = io.imread(A_images + train_img)
    img_B = io.imread(B_images + train_img)

    mask = img_A - img_B
    mask = np.asarray(mask)
    mask[mask != 0] = 1
    mask_vis = mask.copy()
    mask_vis[mask_vis == 1] = 255
    cv.imwrite(save_base_path + train_img, mask)
    cv.imwrite(save_vis_path + train_img, mask_vis)
