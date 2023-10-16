import os
from typing import List

import cv2
import numpy as np
from mindspore.dataset import GeneratorDataset, BatchDataset

from pycocotools.coco import COCO

from segment_anything.dataset.transform import create_transform_pipeline
from segment_anything.utils import logger
from segment_anything.utils.registry import DATASET_REGISTRY


def create_dataloader(args)-> BatchDataset:
    """
    create batched dataloader with specified epochs
    """
    raw_dataset = DATASET_REGISTRY.instantiate(**args.dataset)
    dataloader = GeneratorDataset(source=raw_dataset,
                                 column_names=list(args.dataset.output_column),
                                 num_parallel_workers=args.num_workers,
                                 python_multiprocessing=True,
                                 num_shards=args.rank_size,
                                 shard_id=args.rank_id,
                                 max_rowsize=args.max_rowsize if args.max_rowsize is not None else 6,
                                 )
    dataloader = dataloader.batch(args.batch_size, drop_remainder=args.drop_remainder)

    # repeat dataloader to epoch_size in Model.train instead of here for better compatibility

    return dataloader


@DATASET_REGISTRY.registry_module()
class COCODataset:

    def __init__(self,
                 data_dir,
                 annotation_path,
                 transform_pipeline,
                 output_column: List[str] = None,
                 **kwargs,
                 ):
        self.data_dir = data_dir
        self.transform_pipeline = create_transform_pipeline(transform_pipeline)
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.output_column = output_column

        # Filter out image_ids without any annotation
        logger.info(f'filtering out image_ids without any annotation')
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]
        logger.info(f'coco dateset size: {len(self.image_ids)}')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Returns:
            a tuple of transformed input items
        """
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.data_dir, image_info['file_name'])
        assert os.path.exists(image_path), f'image file not found at {image_path}'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        masks = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        # letter box
        data_dict = dict(image=image, masks=masks, boxes=np.array(boxes, np.float32))
        data_dict = self.transform_pipeline(data_dict)

        if self.output_column is None:
            self.output_column = list(data_dict.key())

        return tuple(data_dict[k] for k in self.output_column)


@DATASET_REGISTRY.registry_module()
class FLAREDataset:

    def __init__(self,
                 data_dir,
                 output_column: List[str],
                 transform_pipeline,
                 **kwargs,
                 ):
        self.data_dir = data_dir
        self.output_column = output_column
        self.transform_pipeline = create_transform_pipeline(transform_pipeline)

        self.img_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        assert os.path.exists(self.img_dir) and os.path.exists(self.label_dir)

        self.img_names = sorted(os.listdir(self.img_dir))
        self.label_names = sorted(os.listdir(self.label_dir))
        assert len(self.img_names) == len(self.label_names)

        for i_n, l_n in zip(self.img_names, self.label_names):
            assert i_n == l_n
        self.dataset_size = len(self.img_names)
        self.img_paths = [os.path.join(self.img_dir, n) for n in self.img_names]
        self.label_paths = [os.path.join(self.label_dir, n) for n in self.label_names]
        logger.info(f'flare dataset size: {self.dataset_size}')

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]

        # Step1. read image
        image = np.load(img_path)
        assert image.shape == (1024, 1024, 3)

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        assert (np.max(image) <= 1.0 and np.min(image) >= 0.0), "image should be normalized to [0, 1]"

        # Step 2. read instance segmentation data
        instance_seg = np.load(label_path)

        # Step3: do transformation, make mask and box by instance seg, pad label to static shape
        data_dict = self.transform_pipeline(data_dict=dict(image=image, instance_seg=instance_seg))

        if self.output_column is None:
            self.output_column = list(data_dict.key())

        return tuple(data_dict[k] for k in self.output_column)
