"""
Create dataloader
"""
import cv2
import random
import multiprocessing
import numpy as np
from numbers import Integral

import mindspore.dataset as de

from utils import logger
from data.utils import xyxy2xywh

__all__ = ["create_loader"]


def bbox_area(src_bbox):
    if src_bbox[2] < src_bbox[0] or src_bbox[3] < src_bbox[1]:
        return 0.
    else:
        width = src_bbox[2] - src_bbox[0]
        height = src_bbox[3] - src_bbox[1]
        return width * height


def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox[0] >= object_bbox[2] or \
        sample_bbox[2] <= object_bbox[0] or \
        sample_bbox[1] >= object_bbox[3] or \
        sample_bbox[3] <= object_bbox[1]:
        return 0
    intersect_xmin = max(sample_bbox[0], object_bbox[0])
    intersect_ymin = max(sample_bbox[1], object_bbox[1])
    intersect_xmax = min(sample_bbox[2], object_bbox[2])
    intersect_ymax = min(sample_bbox[3], object_bbox[3])
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
        sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


class MultiScaleTrans:
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 dataset_size,
                 target_size,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        self.dataset_size = dataset_size
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        self.target_size = target_size
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

        self.size_dict = {}
        self.seed_num = int(1e6)
        self.seed_list = self.generate_seed_list(seed_num=self.seed_num)
        self.resize_rate = 1
        self.resize_count_num = int(np.ceil(self.dataset_size / self.resize_rate))

    def generate_seed_list(self, init_seed=1234, seed_num=int(1e6), seed_range=(1, 1000)):
        seed_list = []
        random.seed(init_seed)
        for _ in range(seed_num):
            seed = random.randint(seed_range[0], seed_range[1])
            seed_list.append(seed)
        return seed_list

    def __call__(self, samples, batch_info):
        # BatchRandomResize
        epoch_num = batch_info.get_epoch_num()
        size_idx = int(batch_info.get_batch_num() / self.resize_rate)
        seed_key = self.seed_list[(epoch_num * self.resize_count_num + size_idx) % self.seed_num]
        if self.size_dict.get(seed_key, None) is None:
            random.seed(seed_key)
            new_size = random.choice(self.target_size)
            self.size_dict[seed_key] = new_size
        seed = seed_key

        target_size = self.size_dict[seed]
        interp = np.random.choice(self.interps)
        resizer = Resize(target_size, keep_ratio=False, interp=interp)
        for sample in samples:
            sample = resizer(sample)

        ## label_norm
        for sample in samples:
            bboxes = sample['bboxes']
            if len(bboxes) == 0:
                continue

            bboxes = xyxy2xywh(bboxes)  # convert xyxy to xywh
            height, width = sample['img'].shape[:2]
            bboxes[:, [1, 3]] /= height  # normalized height 0-1
            bboxes[:, [0, 2]] /= width  # normalized width 0-1
            sample['bboxes'] = bboxes

        ## label_pad
        padding_size = 160
        padding_value = 0.
        for sample in samples:
            bbox_format = sample['bbox_format']
            assert bbox_format == 'xywhn', f'The bbox format should be xywhn, but got {bbox_format}'

            cls, bboxes, scores = sample['cls'], sample['bboxes'], sample['gt_score']
            cls_pad = np.full((padding_size,), padding_value, dtype=np.float32)
            bboxes_pad = np.full((padding_size, 4), padding_value, dtype=np.float32)
            scores_pad = np.full((padding_size,), padding_value, dtype=np.float32)
            nL = len(bboxes)
            if nL:
                cls_pad[:min(nL, padding_size)] = cls[:min(nL, padding_size), 0]
                bboxes_pad[:min(nL, padding_size)] = bboxes[:min(nL, padding_size)]
                scores_pad[:min(nL, padding_size)] = scores[:min(nL, padding_size), 0]
            sample['cls'] = cls_pad
            sample['bboxes'] = bboxes_pad
            sample['gt_score'] = scores_pad

        # normalize_image
        mean = [0.406, 0.456, 0.485]
        std = [0.225, 0.224, 0.229]
        for sample in samples:
            im = sample['img']
            im = im.astype(np.float32, copy=False)
            scale = 1.0 / 255.0
            im *= scale
            mean = np.array(mean)
            std = np.array(std)
            im -= mean
            im /= std
            sample['img'] = im

        # image_transpose
        for sample in samples:
            image = sample['img']
            image = image[:, :, ::-1]
            image = image.transpose(2, 0, 1)
            sample['img'] = image

        ## Gt2YoloTarget
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['img'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            gt_bbox = sample['bboxes']
            gt_class = sample['cls'].astype(np.int32)
            if 'gt_score' not in sample:
                sample['gt_score'] = np.ones(
                    (gt_bbox.shape[0], 1), dtype=np.float32)
            gt_score = sample['gt_score']
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue

                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)

                    # gtbox should be regresed in this layes if best match
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)

                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(
                            gw * w / self.anchors[best_idx][0])
                        target[best_n, 3, gj, gi] = np.log(
                            gh * h / self.anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh

                        # objectness record gt_score
                        target[best_n, 5, gj, gi] = score

                        # classification
                        target[best_n, 6 + cls, gj, gi] = 1.

                    # For non-matched anchors, calculate the target if the iou
                    # between anchor and gt is larger than iou_thresh
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx: continue
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                            if iou > self.iou_thresh and target[idx, 5, gj,
                                                                gi] == 0.:
                                # x, y, w, h, scale
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(
                                    gw * w / self.anchors[mask_i][0])
                                target[idx, 3, gj, gi] = np.log(
                                    gh * h / self.anchors[mask_i][1])
                                target[idx, 4, gj, gi] = 2.0 - gw * gh

                                # objectness record gt_score
                                target[idx, 5, gj, gi] = score

                                # classification
                                target[idx, 6 + cls, gj, gi] = 1.
                sample['target{}'.format(i)] = target

            # remove useless gt_class and gt_score after target calculated
            sample.pop('cls')
            sample.pop('gt_score')

        imgs = [sample.pop('img') for sample in samples]
        target_0 = [sample.pop('target0') for sample in samples]
        target_1 = [sample.pop('target1') for sample in samples]
        target_2 = [sample.pop('target2') for sample in samples]
        gt_bbox = [sample.pop('bboxes') for sample in samples]
        return np.stack(imgs, 0), np.stack(target_0, 0), np.stack(target_1, 0), np.stack(target_2, 0), np.stack(gt_bbox, 0)


def create_loader(
    dataset,
    batch_collate_fn,
    column_names_getitem,
    column_names_collate,
    batch_size,
    epoch_size=1,
    rank=0,
    rank_size=1,
    num_parallel_workers=8,
    shuffle=True,
    drop_remainder=False,
    python_multiprocessing=False,
):
    r"""Creates dataloader.

    Applies operations such as transform and batch to the `ms.dataset.Dataset` object
    created by the `create_dataset` function to get the dataloader.

    Args:
        dataset (COCODataset): dataset object created by `create_dataset`.
        batch_size (int or function): The number of rows each batch is created with. An
            int or callable object which takes exactly 1 parameter, BatchInfo.
        drop_remainder (bool, optional): Determines whether to drop the last block
            whose data row number is less than batch size (default=False). If True, and if there are less
            than batch_size rows available to make the last batch, then those rows will
            be dropped and not propagated to the child node.
        num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel
            (default=None).
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker processes. This
            option could be beneficial if the Python operation is computational heavy (default=False).

    Returns:
        BatchDataset, dataset batched.
    """
    de.config.set_seed(1236517205 + rank)
    cores = multiprocessing.cpu_count()
    num_parallel_workers = min(int(cores / rank_size), num_parallel_workers)
    logger.info(f"Dataloader num parallel workers: [{num_parallel_workers}]")
    if rank_size > 1:
        ds = de.GeneratorDataset(
            dataset,
            column_names=column_names_getitem,
            num_parallel_workers=min(8, num_parallel_workers),
            shuffle=shuffle,
            python_multiprocessing=python_multiprocessing,
            num_shards=rank_size,
            shard_id=rank,
        )
    else:
        ds = de.GeneratorDataset(
            dataset,
            column_names=column_names_getitem,
            num_parallel_workers=min(32, num_parallel_workers),
            shuffle=shuffle,
            python_multiprocessing=python_multiprocessing,
        )
    ds = ds.batch(
        batch_size, per_batch_map=batch_collate_fn,
        input_columns=column_names_getitem, output_columns=column_names_collate, drop_remainder=drop_remainder
    )
    ds = ds.repeat(epoch_size)

    return ds


class Resize:
    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        self.keep_ratio = keep_ratio
        self.interp = interp
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox

    def __call__(self, sample):
        """ Resize the image numpy.
        """
        im = sample['img']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))

        # apply image
        if len(im.shape) == 3:
            im_shape = im.shape
        else:
            im_shape = im[0].shape

        resize_h, resize_w = self.target_size

        im_scale_y = resize_h / im_shape[0]
        im_scale_x = resize_w / im_shape[1]

        if len(im.shape) == 3:
            im = self.apply_image(sample['img'], [im_scale_x, im_scale_y])
            sample['img'] = im.astype(np.float32)
        else:
            resized_images = []
            for one_im in im:
                applied_im = self.apply_image(one_im, [im_scale_x, im_scale_y])
                resized_images.append(applied_im)

            sample['img'] = np.array(resized_images)

        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'bboxes' in sample and len(sample['bboxes']) > 0:
            sample['bboxes'] = self.apply_bbox(sample['bboxes'],
                                                [im_scale_x, im_scale_y],
                                                [resize_w, resize_h])

        return sample
