# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""
Data operations, will be used in train.py and eval.py
"""
import os
import math

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter
from mindspore.communication.management import get_rank, get_group_size

from src.data.augment.auto_augment import pil_interp, rand_augment_transform
from src.data.augment.mixup import Mixup
from src.data.augment.random_erasing import RandomErasing

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def create_val_dataset(
        data_path,
        val_dir,
        num_workers,
        input_size,
        eval_crop_ratio,
        batch_size,
        mixup,
        cutmix,
        num_classes,
        interpolation,
        repeat_num=1
):
    val_dir = os.path.join(data_path, val_dir)

    data_set = ds.ImageFolderDataset(
        val_dir,
        num_parallel_workers=num_workers,
        shuffle=False
    )

    transform_img = get_validation_transforms(
        input_size, interpolation=interpolation, crop_pct=eval_crop_ratio
    )
    transform_label = transforms.TypeCast(mstype.int32)

    data_set = data_set.map(
        input_columns="image",
        num_parallel_workers=num_workers,
        operations=transform_img
    )
    data_set = data_set.map(
        input_columns="label",
        num_parallel_workers=num_workers,
        operations=transform_label
    )
    if mixup > 0.0 or cutmix > 0.0:
        # if use mixup and not training(False), one hot val data label
        one_hot = transforms.OneHot(num_classes=num_classes)
        cast_type = transforms.TypeCast(mstype.float32)
        data_set = data_set.map(
            input_columns=["label"],
            num_parallel_workers=num_workers,
            operations=[one_hot, cast_type],
        )
    # apply batch operations
    data_set = data_set.batch(
        batch_size,
        drop_remainder=False,
        num_parallel_workers=num_workers
    )

    data_set = data_set.repeat(repeat_num)

    return data_set


def create_train_dataset(
        data_path,
        train_dir,
        num_parallel_workers,
        input_size,
        interpolation,
        auto_augment,
        min_crop,
        reprob,
        remode,
        recount,
        batch_size,
        mixup,
        cutmix,
        cutmix_minmax,
        mixup_prob,
        mixup_switch_prob,
        mixup_mode,
        smoothing,
        num_classes,
        mixup_off_epoch,
        repeat_num=1
):
    device_num, rank_id = _get_rank_info()
    shuffle = True
    drop_remainder = True

    dataset_dir = os.path.join(data_path, train_dir)
    if device_num == 1:
        data_set = ds.ImageFolderDataset(
            dataset_dir,
            num_parallel_workers=num_parallel_workers,
            shuffle=shuffle
        )
    else:

        data_set = ds.ImageFolderDataset(
            dataset_dir,
            num_parallel_workers=num_parallel_workers,
            shuffle=shuffle,
            num_shards=device_num,
            shard_id=rank_id
        )


    transform_img = get_train_transforms(
        input_size, interpolation, auto_augment, min_crop,
        reprob, remode, recount
    )

    transform_label = transforms.TypeCast(mstype.int32)

    data_set = data_set.map(
        input_columns="image",
        num_parallel_workers=num_parallel_workers,
        operations=transform_img
    )
    data_set = data_set.map(
        input_columns="label",
        num_parallel_workers=num_parallel_workers,
        operations=transform_label
    )

    # apply batch operations
    data_set = data_set.batch(
        batch_size,
        drop_remainder=drop_remainder,
        num_parallel_workers=num_parallel_workers
    )

    if (mixup > 0.0 or cutmix > 0.0):
        mixup_fn = Mixup(
            mixup_alpha=mixup,
            cutmix_alpha=cutmix,
            cutmix_minmax=cutmix_minmax,
            prob=mixup_prob,
            switch_prob=mixup_switch_prob,
            mode=mixup_mode,
            label_smoothing=smoothing,
            num_classes=num_classes,
            mix_steps=int(mixup_off_epoch * data_set.get_dataset_size())
        )

        data_set = data_set.map(
            operations=mixup_fn,
            input_columns=["image", "label"],
            num_parallel_workers=num_parallel_workers
        )

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def create_datasets(args):
    val_dataset = create_val_dataset(
        args.data_path,
        args.val_dir,
        args.num_workers,
        args.input_size,
        args.eval_crop_ratio,
        args.batch_size,
        args.mixup,
        args.cutmix,
        args.num_classes,
        args.interpolation
    )

    train_dataset = create_train_dataset(
        args.data_path,
        args.train_dir,
        args.num_workers,
        args.input_size,
        args.interpolation,
        args.aa,
        args.min_crop,
        args.reprob,
        args.remode,
        args.recount,
        args.batch_size,
        args.mixup,
        args.cutmix,
        args.cutmix_minmax,
        args.mixup_prob,
        args.mixup_switch_prob,
        args.mixup_mode,
        args.smoothing,
        args.num_classes,
        args.mixup_off_epoch,
        repeat_num=1
    )

    return train_dataset, val_dataset


def get_validation_transforms(
        image_size,
        interpolation,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        crop_pct=None,
):
    """Get only validation transforms."""
    crop_pct = crop_pct or 0.875
    scale_size = int(math.floor(image_size / crop_pct))

    transform_img = [
        vision.Decode(),
        vision.Resize(scale_size, interpolation=Inter.PILCUBIC),
        vision.CenterCrop(image_size),
        vision.ToTensor(),
        vision.Normalize(mean=mean, std=std, is_hwc=False),
    ]
    return transform_img


def get_train_transforms(
        image_size,
        interpolation,
        auto_augment,
        min_crop,
        re_prob,
        re_mode,
        re_count,
):
    """Get main transform"""

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    aa_params = dict(
        translate_const=int(image_size * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in mean]),
    )
    interpolation = interpolation
    auto_augment = auto_augment
    aa_params['interpolation'] = pil_interp(interpolation)

    assert 0 <= min_crop < 1
    transform_img = [
        vision.RandomCropDecodeResize(
            image_size,
            scale=(min_crop, 1.0),
            ratio=(3 / 4, 4 / 3),
            interpolation=Inter.PILCUBIC
        ),
        vision.RandomHorizontalFlip(prob=0.5),
        vision.ToPIL()
    ]
    if auto_augment != "None":
        transform_img += [rand_augment_transform(auto_augment, aa_params)]
    transform_img += [
        vision.ToTensor(),
        vision.Normalize(mean=mean, std=std, is_hwc=False),
        RandomErasing(re_prob, mode=re_mode, max_count=re_count)
    ]

    return transform_img


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
