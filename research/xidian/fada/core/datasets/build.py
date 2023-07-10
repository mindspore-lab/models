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
import mindspore
from mindspore.dataset.vision import c_transforms as c_vision
from .dataset_path_catalog import DatasetCatalog
from . import transform


def build_transform(cfg, mode, is_source, flag=True):
    if mode=="train":
        w, h = cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN if is_source else cfg.INPUT.TARGET_INPUT_SIZE_TRAIN
        trans_list = [
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ]
        if cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN > 0:
            trans_list = [transform.RandomHorizontalFlip(p=cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN),] + trans_list
            
        if is_source:
            trans_list = [
                transform.ColorJitter(
                    brightness=cfg.INPUT.BRIGHTNESS,
                    contrast=cfg.INPUT.CONTRAST,
                    saturation=cfg.INPUT.SATURATION,
                    hue=cfg.INPUT.HUE,
                ),
            ] + trans_list
        
        if cfg.INPUT.INPUT_SCALES_TRAIN[0]==cfg.INPUT.INPUT_SCALES_TRAIN[1] and cfg.INPUT.INPUT_SCALES_TRAIN[0]==1:
            trans_list = [transform.Resize((h, w)),] + trans_list
        else:
            trans_list = [
                transform.RandomScale(scale=cfg.INPUT.INPUT_SCALES_TRAIN),
                transform.RandomCrop(size=(h, w), pad_if_needed=True),
            ] + trans_list
        
        
        trans = transform.Compose(trans_list)
    else:
        w, h = cfg.INPUT.INPUT_SIZE_TEST
        trans = transform.Compose([
            transform.Resize((h, w), resize_label=False),
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ])
    return trans


def build_dataset(cfg, mode='train', is_source=True, epochwise=False, flag=True):
    assert mode in ['train', 'val', 'test']
    transform = build_transform(cfg, mode, is_source, flag)
    iters = None
    if mode=='train':
        if not epochwise:
            iters = cfg.SOLVER.MAX_ITER*cfg.SOLVER.BATCH_SIZE
        if is_source:
            dataset = DatasetCatalog.get(cfg.DATASETS.SOURCE_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=transform)
        else:
            dataset = DatasetCatalog.get(cfg.DATASETS.TARGET_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=transform)
    elif mode=='val':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, 'val', num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=transform)
    elif mode=='test':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, cfg.DATASETS.TEST.split('_')[-1], num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=transform)

    return dataset
