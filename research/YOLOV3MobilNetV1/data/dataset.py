import os
import cv2
import hashlib
import random
import glob
import numpy as np
from pathlib import Path
from PIL import ExifTags, Image
from tqdm import tqdm
from copy import deepcopy

from utils import logger

from data.utils import segments2boxes, xywhn2xyxy

__all__ = ["COCODataset"]


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


class COCODataset:
    """
    Load the COCO dataset (yolo format coco labels)

    Args:
        dataset_path (str): dataset label directory for dataset.
        for example:
            COCO_ROOT
                ├── train2017.txt
                ├── annotations
                │     └── instances_train2017.json
                ├── images
                │     └── train2017
                │             ├── 000000000001.jpg
                │             └── 000000000002.jpg
                └── labels
                      └── train2017
                              ├── 000000000001.txt
                              └── 000000000002.txt
            dataset_path (str): ./coco/train2017.txt
        transforms (list): A list of images data enhancements
            that apply data enhancements on data set objects in order.
    """

    def __init__(
        self,
        dataset_path="",
        img_size=640,
        transforms_dict=None,
        is_training=False,
        augment=False,
        rect=False,
        single_cls=False,
        batch_size=32,
        stride=32,
        num_cls=80,
        pad=0.0,
        return_segments=False,  # for segment
        return_keypoints=False, # for keypoint
        nkpt=0,                 # for keypoint
        ndim=0                  # for keypoint
    ):
        # acceptable image suffixes
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
        self.cache_version = 0.2

        self.return_segments = return_segments
        self.return_keypoints = return_keypoints
        assert not (return_segments and return_keypoints), 'Can not return both segments and keypoints.'

        self.path = dataset_path
        self.img_size = img_size
        self.augment = augment
        self.rect = rect
        self.stride = stride
        self.num_cls = num_cls
        self.nkpt = nkpt
        self.ndim = ndim
        self.transforms_dict = transforms_dict
        self.is_training = is_training

        # set column names
        self.column_names_getitem = ['samples']
        if self.is_training:
            self.column_names_collate = ['images', 'target0', 'target1', 'target2', 'gt_bbox']
        else:
            self.column_names_collate = ["images", "img_files", "hw_ori", "hw_scale", "pad"]

        try:
            f = []  # image files
            for p in self.path if isinstance(self.path, list) else [self.path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                elif p.is_file():  # file
                    with open(p, "r") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                else:
                    raise Exception(f"{p} does not exist")
            self.img_files = sorted([x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in self.img_formats])
            assert self.img_files, f"No images found"
        except Exception as e:
            raise Exception(f"Error loading data from {self.path}: {e}\n")

        # Check cache
        self.label_files = self._img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache.npy")  # cached labels
        if cache_path.is_file():
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            if cache["version"] == self.cache_version \
                    and cache["hash"] == self._get_hash(self.label_files + self.img_files):
                logger.info(f"Dataset Cache file hash/version check success.")
                logger.info(f"Load dataset cache from [{cache_path}] success.")
            else:
                logger.info(f"Dataset cache file hash/version check fail.")
                logger.info(f"Datset caching now...")
                cache, exists = self.cache_labels(cache_path), False  # cache
                logger.info(f"Dataset caching success.")
        else:
            logger.info(f"No dataset cache available, caching now...")
            cache, exists = self.cache_labels(cache_path), False  # cache
            logger.info(f"Dataset caching success.")

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f"No labels in {cache_path}. Can not train without labels. See {self.help_url}"

        # Read cache
        cache.pop("hash")  # remove hash
        cache.pop("version")  # remove version
        self.labels = cache['labels']
        self.img_files = [lb['im_file'] for lb in self.labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in self.labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            print(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in self.labels:
                lb['segments'] = []
        if len_cls == 0:
            raise ValueError(f'All labels empty in {cache_path}, can not start training without labels.')

        if single_cls:
            for x in self.labels:
                x['cls'][:, 0] = 0

        n = len(self.labels)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int_)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_hw_ori, self.indices = None, None, range(n)

        # Rectangular Train/Test
        if self.rect:
            # Sort by aspect ratio
            s = self.img_shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.img_shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        self.imgIds = [int(Path(im_file).stem) for im_file in self.img_files]

    def cache_labels(self, path=Path("./labels.cache.npy")):
        # Cache dataset labels, check images and read shapes
        x = {'labels': []}  # dict
        nm, nf, ne, nc, segments, keypoints = 0, 0, 0, 0, [], None  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc="Scanning images", total=len(self.img_files))
        if self.return_keypoints and (self.nkpt <= 0 or self.ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = self._exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
                assert im.format.lower() in self.img_formats, f"invalid image format {im.format}"

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, "r") as f:
                        lb = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 6 for x in lb]) and (not self.return_keypoints):  # is segment
                            classes = np.array([x[0] for x in lb], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                            lb = np.concatenate(
                                (classes.reshape(-1, 1), segments2boxes(segments)), 1
                            )  # (cls, xywh)
                        lb = np.array(lb, dtype=np.float32)
                    nl = len(lb)
                    if nl:
                        if self.return_keypoints:
                            assert lb.shape[1] == (5 + self.nkpt * self.ndim), \
                                f'labels require {(5 + self.nkpt * self.ndim)} columns each'
                            assert (lb[:, 5::self.ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                            assert (lb[:, 6::self.ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        else:
                            assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                            assert (lb[:, 1:] <= 1).all(), \
                                f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                            assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                        # All labels
                        max_cls = int(lb[:, 0].max())  # max label count
                        assert max_cls <= self.num_cls, \
                            f'Label class {max_cls} exceeds dataset class count {self.num_cls}. ' \
                            f'Possible class labels are 0-{self.num_cls - 1}'
                        _, j = np.unique(lb, axis=0, return_index=True)
                        if len(j) < nl:  # duplicate row check
                            lb = lb[j]  # remove duplicates
                            if segments:
                                segments = [segments[x] for x in i]
                            print(f'WARNING ⚠️ {im_file}: {nl - len(j)} duplicate labels removed')
                    else:
                        ne += 1  # label empty
                        lb = np.zeros((0, (5 + self.nkpt * self.ndim)), dtype=np.float32) \
                            if self.return_keypoints else np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    lb = np.zeros((0, (5 + self.nkpt * self.ndim)), dtype=np.float32) \
                        if self.return_keypoints else np.zeros((0, 5), dtype=np.float32)
                if self.return_keypoints:
                    keypoints = lb[:, 5:].reshape(-1, self.nkpt, self.ndim)
                    if self.ndim == 2:
                        kpt_mask = np.ones(keypoints.shape[:2], dtype=np.float32)
                        kpt_mask = np.where(keypoints[..., 0] < 0, 0.0, kpt_mask)
                        kpt_mask = np.where(keypoints[..., 1] < 0, 0.0, kpt_mask)
                        keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
                lb = lb[:, :5]
                x['labels'].append(
                    dict(
                        im_file=im_file,
                        cls=lb[:, 0:1],     # (n, 1)
                        bboxes=lb[:, 1:],   # (n, 4)
                        segments=segments,  # list of (mi, 2)
                        keypoints=keypoints,
                        bbox_format='xywhn',
                        segment_format='polygon'
                    )
                )
            except Exception as e:
                nc += 1
                print(f"WARNING: Ignoring corrupted image and/or label {im_file}: {e}")

            pbar.desc = f"Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f"WARNING: No labels found in {path}. See {self.help_url}")

        x["hash"] = self._get_hash(self.label_files + self.img_files)
        x["results"] = nf, nm, ne, nc, len(self.img_files)
        x["version"] = self.cache_version  # cache version
        np.save(path, x)  # save for next time
        logger.info(f"New cache created: {path}")
        return x

    def __getitem__(self, index):
        sample = self.get_sample(index)

        for _i, ori_trans in enumerate(self.transforms_dict):
            _trans = ori_trans.copy()
            func_name, prob = _trans.pop("func_name"), _trans.pop("prob", 1.0)
            if func_name == 'copy_paste':
                sample = self.copy_paste(sample, prob)
            elif random.random() < prob:
                if func_name == "letterbox":
                    new_shape = self.img_size if not self.rect else self.batch_shapes[self.batch[index]]
                    sample = self.letterbox(sample, new_shape, **_trans)
                else:
                    sample = getattr(self, func_name)(sample, **_trans)

        sample['img'] = np.ascontiguousarray(sample['img'])
        return sample

    def __len__(self):
        return len(self.img_files)

    def get_sample(self, index):
        """Get and return label information from the dataset."""
        sample = deepcopy(self.labels[index])
        if self.imgs is None:
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR
            assert img is not None, "Image Not Found " + path
            h_ori, w_ori = img.shape[:2]  # orig hw
            r = self.img_size / max(h_ori, w_ori)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)

            sample['img'], sample['ori_shape'] = img, np.array([h_ori, w_ori])  # img, hw_original

        else:
            sample['img'], sample['ori_shape'] = self.imgs[index], self.img_hw_ori[index]  # img, hw_original

        return sample

    def resample_segments(self, sample, n=1000):
        segment_format = sample['segment_format']
        assert segment_format == 'polygon', f'The segment format is should be polygon, but got {segment_format}'

        segments = sample['segments']
        if len(segments) > 0:
            # Up-sample an (n,2) segment
            for i, s in enumerate(segments):
                s = np.concatenate((s, s[0:1, :]), axis=0)
                x = np.linspace(0, len(s) - 1, n)
                xp = np.arange(len(s))
                segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
            segments = np.stack(segments, axis=0)
        else:
            segments = np.zeros((0, 1000, 2), dtype=np.float32)
        sample['segments'] = segments
        return sample

    def mixup(self, sample, alpha: 32.0, beta: 32.0):
        index = random.choices(self.indices, k=1)[0]
        sample2 = self.get_sample(index)

        image, image2 = sample['img'], sample2['img']
        (h1, w1) = image.shape[:2]
        (h2, w2) = image2.shape[:2]
        r = np.random.beta(alpha, beta)  # mixup ratio, alpha=beta=8.0
        h = max(h1, h2)
        w = max(w1, w2)
        img = np.zeros((h, w, image.shape[2]), 'float32')
        img[:image.shape[0], :image.shape[1], :] = \
            image.astype('float32') * r
        img[:image2.shape[0], :image2.shape[1], :] += \
            image2.astype('float32') * (1.0 - r)

        gt_score1 = np.ones_like(sample['cls'])
        gt_score2 = np.ones_like(sample2['cls'])

        bboxes1, bboxes2 = sample['bboxes'], sample2['bboxes']
        bboxes1 = xywhn2xyxy(bboxes1, w1, h1)
        bboxes2 = xywhn2xyxy(bboxes2, w2, h2)
        sample['img'] = img
        sample['cls'] = np.concatenate((sample['cls'], sample2['cls']), 0)
        sample['bboxes'] = np.concatenate((bboxes1, bboxes2), 0)

        gt_score = np.concatenate((gt_score1 * r, gt_score2 * (1. - r)), axis=0)
        sample['gt_score'] = gt_score.astype('float32')
        return sample

    def fliplr(self, sample):
        # flip image left-right
        image = sample['img']
        image = np.fliplr(image)
        sample['img'] = image

        # flip box
        _, w = image.shape[:2]
        bboxes, bbox_format = sample['bboxes'], sample['bbox_format']

        if len(bboxes):
            x1 = bboxes[:, 0].copy()
            x2 = bboxes[:, 2].copy()
            bboxes[:, 0] = w - x2
            bboxes[:, 2] = w - x1

        sample['bboxes'] = bboxes

        return sample

    def normalize_image(self, sample, mean=[0, 0, 0], std=[1, 1, 1]):
        im = sample['img']
        im = im.astype(np.float32, copy=False)
        scale = 1.0 / 255.0
        im *= scale
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im -= mean
        im /= std
        sample['img'] = im
        return sample

    def resize(self, sample, target_size=[608, 608]):
        im = sample['img']
        im_shape = im.shape
        resize_h, resize_w = target_size
        im_scale_y = resize_h / im_shape[0]
        im_scale_x = resize_w / im_shape[1]

        ori_shape = sample['ori_shape']
        h, w = target_size
        h0, w0 = ori_shape
        hw_scale = np.array([h / h0, w / w0])
        sample['hw_scale'] = hw_scale

        hw_pad = np.array([0., 0.])
        sample['hw_pad'] = hw_pad

        sample['img'] = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=2)
        return sample

    def image_transpose(self, sample, bgr2rgb=True, hwc2chw=True):
        image = sample['img']
        if bgr2rgb:
            image = image[:, :, ::-1]
        if hwc2chw:
            image = image.transpose(2, 0, 1)
        sample['img'] = image
        return sample

    def random_distort(self, sample, random_apply=True, count=4, random_channel=False,
                       hue=[-18, 18, 0.5], saturation=[0.5, 1.5, 0.5], contrast=[0.5, 1.5, 0.5], brightness=[0.5, 1.5, 0.5]):
        img = sample['img']
        if random_apply:
            functions = [
                distort_brightness, distort_contrast,
                distort_saturation, distort_hue
            ]
            args = [brightness, contrast, saturation, hue]
            distortions = np.random.permutation(functions)[:count]
            for func, arg in zip(distortions, args):
                img = func(img, arg)
            sample['img'] = img
            return sample

        img = distort_brightness(img)
        mode = np.random.randint(0, 2)

        if mode:
            img = distort_contrast(img, contrast)

        img = distort_saturation(img, saturation)
        img = distort_hue(img, hue)

        if not mode:
            img = distort_contrast(img, contrast)

        if random_channel:
            if np.random.randint(0, 2):
                img = img[..., np.random.permutation(3)]
        sample['img'] = img
        return sample

    def random_expand(self, sample, ratio=4., fill_value=(127.5, 127.5, 127.5)):
        im = sample['img']
        height, width = im.shape[:2]
        ratio = np.random.uniform(1., ratio)
        h = int(height * ratio)
        w = int(width * ratio)
        if not h > height or not w > width:
            return sample
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        offsets, size = [x, y], [h, w]

        return self.pad(sample, size, pad_mode=-1, offsets=offsets, fill_value=fill_value)

    def pad(self, sample, size=None, size_divisor=32, pad_mode=0, offsets=None, fill_value=(127.5, 127.5, 127.5)):
        im = sample['img']
        im_h, im_w = im.shape[:2]
        if size:
            h, w = size
            assert (
                    im_h <= h and im_w <= w
            ), '(h, w) of target size should be greater than (im_h, im_w)'
        else:
            h = int(np.ceil(im_h / size_divisor) * size_divisor)
            w = int(np.ceil(im_w / size_divisor) * size_divisor)

        if h == im_h and w == im_w:
            sample['img'] = im.astype(np.float32)
            return sample

        if pad_mode == -1:
            offset_x, offset_y = offsets
        elif pad_mode == 0:
            offset_y, offset_x = 0, 0
        elif pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(fill_value, dtype=np.float32)
        canvas[offset_y:offset_y + im_h, offset_x:offset_x + im_w, :] = im.astype(np.float32)
        sample['img'] = canvas

        if pad_mode == 0:
            return sample
        if 'bboxes' in sample and len(sample['bboxes']) > 0:
            sample['bboxes'] += np.array(offsets * 2, dtype=np.float32)

        return sample

    def random_crop(self, sample, aspect_ratio=(0.5, 2.0), thresholds=[.0, .1, .3, .5, .7, .9], scaling=[.3, 1.],
                    num_attempts=50, allow_no_crop=True, cover_all_box=False, fake_bboxes=False):
        h, w = sample['img'].shape[:2]
        gt_bbox = sample['bboxes']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        if allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return sample

            found = False
            for i in range(num_attempts):
                scale = np.random.uniform(*scaling)
                if aspect_ratio is not None:
                    min_ar, max_ar = aspect_ratio
                    aspect_ratio_ = np.random.uniform(
                        max(min_ar, scale ** 2), min(max_ar, scale ** -2))
                    h_scale = scale / np.sqrt(aspect_ratio_)
                    w_scale = scale * np.sqrt(aspect_ratio_)
                else:
                    h_scale = np.random.uniform(*scaling)
                    w_scale = np.random.uniform(*scaling)
                crop_h = h * h_scale
                crop_w = w * w_scale
                if aspect_ratio is None:
                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                        continue

                crop_h = int(crop_h)
                crop_w = int(crop_w)
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                x1, y1, x2, y2 = crop_box
                sample['img'] = sample['img'][y1:y2, x1:x2, :]
                if fake_bboxes == True:
                    return sample

                sample['bboxes'] = np.take(cropped_box, valid_ids, axis=0)
                sample['cls'] = np.take(
                    sample['cls'], valid_ids, axis=0)
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)

                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _img2label_paths(self, img_paths):
        # Define label paths as a function of image paths
        sa, sb = os.sep + "images" + os.sep, os.sep + "labels" + os.sep  # /images/, /labels/ substrings
        return ["txt".join(x.replace(sa, sb, 1).rsplit(x.split(".")[-1], 1)) for x in img_paths]

    def _get_hash(self, paths):
        # Returns a single hash value of a list of paths (files or dirs)
        size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
        h = hashlib.md5(str(size).encode())  # hash sizes
        h.update("".join(paths).encode())  # hash paths
        return h.hexdigest()  # return hash

    def _exif_size(self, img):
        # Returns exif-corrected PIL size
        s = img.size  # (width, height)
        try:
            rotation = dict(img._getexif().items())[orientation]
            if rotation == 6:  # rotation 270
                s = (s[1], s[0])
            elif rotation == 8:  # rotation 90
                s = (s[1], s[0])
        except:
            pass

        return s

    def train_collate_fn(self, batch_samples, batch_info):
        imgs = [sample.pop('img') for sample in batch_samples]
        labels = []
        for i, sample in enumerate(batch_samples):
            cls, bboxes = sample.pop('cls'), sample.pop('bboxes')
            labels.append(np.concatenate((np.full_like(cls, i), cls, bboxes), axis=-1))
        return_items = [np.stack(imgs, 0), np.stack(labels, 0)]

        if self.return_segments:
            masks = [sample.pop('segments', None) for sample in batch_samples]
            return_items.append(np.stack(masks, 0))
        if self.return_keypoints:
            keypoints = [sample.pop('keypoints', None) for sample in batch_samples]
            return_items.append(np.stack(keypoints, 0))

        return return_items

    def test_collate_fn(self, batch_samples, batch_info):
        imgs = [sample.pop('img') for sample in batch_samples]
        path = [sample.pop('im_file') for sample in batch_samples]
        hw_ori = [sample.pop('ori_shape') for sample in batch_samples]
        hw_scale = [sample.pop('hw_scale') for sample in batch_samples]
        pad = [sample.pop('hw_pad') for sample in batch_samples]
        return (
            np.stack(imgs, 0),
            path,
            np.stack(hw_ori, 0),
            np.stack(hw_scale, 0),
            np.stack(pad, 0),
        )


def distort_hue(img, hue):
    low, high, prob = hue
    if np.random.uniform(0., 1.) < prob:
        return img

    img = img.astype(np.float32)
    # it works, but result differ from HSV version
    delta = np.random.uniform(low, high)
    u = np.cos(delta * np.pi)
    w = np.sin(delta * np.pi)
    bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
    tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                     [0.211, -0.523, 0.311]])
    ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                      [1.0, -1.107, 1.705]])
    t = np.dot(np.dot(ityiq, bt), tyiq).T
    img = np.dot(img, t)
    return img


def distort_saturation(img, saturation):
    low, high, prob = saturation
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)
    img = img.astype(np.float32)
    # it works, but result differ from HSV version
    gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
    gray = gray.sum(axis=2, keepdims=True)
    gray *= (1.0 - delta)
    img *= delta
    img += gray
    return img


def distort_contrast(img, contrast):
    low, high, prob = contrast
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)
    img = img.astype(np.float32)
    img *= delta
    return img


def distort_brightness(img, brightness):
    low, high, prob = brightness
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)
    img = img.astype(np.float32)
    img += delta
    return img
