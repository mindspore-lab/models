import random
from typing import Tuple, Dict, List

import numpy as np

from segment_anything.utils.registry import TRANSFORM_REGISTRY
from segment_anything.utils.transforms import ResizeLongestSide


class TransformPipeline:
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, data_dict):
        for p in self.transform_list:
            data_dict = p(data_dict)
        return data_dict


def create_transform_pipeline(args: List):
    """
    instantiate data transform class and return a callable function that can run the pipeline
    """

    transform_list = []
    for trans in args:
        transform_list.append(TRANSFORM_REGISTRY.instantiate(**trans))

    transform_pipeline = TransformPipeline(transform_list)

    return transform_pipeline


@TRANSFORM_REGISTRY.registry_module()
class ImageNorm:
    def __init__(self,
                 pixel_mean: Tuple[float] = (123.675, 116.28, 103.53),
                 pixel_std: Tuple[float] = (58.395, 57.12, 57.375),
                 hwc2chw: bool = True,
                 ):
        """
        Normalize an image with mean and std value. Convert chanel to chw if asked.
        Args:
            pixel_mean (Tuple[float]): pixel mean value in [0, 255]
            pixel_std (Tuple[float]): pixel std value in [0, 255]
        """
        self.pixel_mean = np.array(pixel_mean, np.float32)
        self.pixel_std = np.array(pixel_std, np.float32)
        self.hwc2chw = hwc2chw

    def __call__(self, result_dict):
        """
        Norm an image with given mean and std, also adjust transpose the chanel when specified.

        Required keys: image, image_pad_area
        Updated keys: image
        Added keys:
        """
        x = result_dict['image']  # (h, w, c)
        x = (x - self.pixel_mean) / self.pixel_std

        # recover padding area to 0
        h, w, _ = x.shape
        pad_value = 0.0
        (pad_h_left, pad_h_right), (pad_w_left, pad_w_right) = result_dict['image_pad_area']
        if pad_h_left > 0:
            x[:pad_h_left] = pad_value
        if pad_h_right > 0:
            x[-pad_h_right:] = pad_value
        if pad_w_left > 0:
            x[:, :pad_w_left] = pad_value
        if pad_w_right > 0:
            x[:, -pad_w_right:] = pad_value

        if self.hwc2chw:
            x = np.transpose(x, (2, 0, 1))
        if False:  # show image and mask for debug
            import matplotlib.pyplot as plt
            plt.imshow(result_dict['image'])  # raw image
            from use_sam_with_promts import show_box, show_mask
            show_box(result_dict['boxes'][0], plt.gca())
            show_mask(result_dict['masks'][0], plt.gca())
            plt.show()

            plt.imshow(x.transpose([1,2,0]))  # normed image
            plt.show()
        result_dict['image'] = x
        return result_dict


@TRANSFORM_REGISTRY.registry_module()
class LabelPad:
    def __init__(self,
                 gt_size=64,
                 to_numpy=True,
                 ):
        """
        Args：
            gt_size (int): gt size for box and mask to pad length to. If object number in an image is less than gt_size,
                    it will be padded to gt_size. If object number in an image is more than gt_size,
                    a number of gt_size objects will be randomly picked out and extra objects will be discarded.
        """
        self.gt_size = gt_size
        self.to_numpy = to_numpy

    def __call__(self, result_dict):
        """
        Pad the label to the given gt_size for the static shape

        Required keys: masks, boxes
        update keys: masks, boxes
        add keys: valid_boxes
        """

        boxes = result_dict['boxes']
        masks = result_dict['masks']
        # Pad box and mask to a fixed length
        valid_gt_size = len(boxes)
        valid_boxes = [1] * self.gt_size
        if valid_gt_size > self.gt_size:
            masks = masks[:self.gt_size]
            boxes = boxes[:self.gt_size]
        elif valid_gt_size < self.gt_size:
            pad_len = self.gt_size - valid_gt_size
            h, w  = masks[0].shape
            masks = np.concatenate([masks, np.zeros((pad_len, h, w), np.uint8)], axis=0)
            boxes= np.concatenate([boxes, np.zeros((pad_len, 4), np.float32)], axis=0)
            valid_boxes = [1] * valid_gt_size + [0] * pad_len

        result_dict['boxes'] = boxes  # (n, 4)
        result_dict['masks'] = masks  # (n, h, w)
        result_dict['valid_boxes'] = np.array(valid_boxes, dtype=np.uint8)
        return result_dict


@TRANSFORM_REGISTRY.registry_module()
class ImageResizeAndPad:

    def __init__(self, target_size):
        """
        Args:
            target_size (int): target size of model input (1024 in sam)
        """
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)

    def __call__(self, result_dict):
        """
        Resize input to the long size and then pad it to the model input size (1024*1024 in sam).
        Pad masks and boxes to a fixed length for graph mode
        Required keys: image, masks, boxes
        Update keys: image, masks, boxes
        Add keys:
            origin_hw (np.array): array with shape (4), represents original image height, width
            and resized height, width, respectively. This array record the trace of image shape transformation
            and is used for visualization.
            image_pad_area (Tuple): image padding area in h and w direction, in the format of
            ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right))
        """

        image = result_dict['image']
        masks = result_dict['masks']
        boxes = result_dict['boxes']

        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        resized_h, resized_w, _ = image.shape
        masks = np.stack([self.transform.apply_image(mask) for mask in masks])

        # Pad image and masks to the model input
        h, w, c = image.shape
        max_dim = max(h, w)  # long side length
        assert max_dim == self.target_size
        # pad 0 to the right and bottom side
        pad_h = max_dim - h
        pad_w = max_dim - w
        img_padding = ((0, pad_h), (0, pad_w), (0, 0))
        image = np.pad(image, pad_width=img_padding, constant_values=0)  # (h, w, c)
        mask_padding = ((0, 0), (0, pad_h), (0, pad_w))  # (n, h, w)
        masks = np.pad(masks, pad_width=mask_padding, constant_values=0)

        # Adjust bounding boxes
        boxes = self.transform.apply_boxes(boxes, (og_h, og_w)).astype(np.float32)

        result_dict['origin_hw'] = np.array([og_h, og_w, resized_h, resized_w], np.int32)  # record image shape trace for visualization
        result_dict['image'] = image
        result_dict['masks'] = masks
        result_dict['boxes'] = boxes
        result_dict['image_pad_area'] = img_padding[:2]

        return result_dict


@TRANSFORM_REGISTRY.registry_module()
class BinaryMaskFromInstanceSeg():
    def __init__(self, ignore_id=0, specify_id=None):
        self.ignore_id = None
        if ignore_id is not None:
            self.ignore_id = ignore_id if isinstance(ignore_id, (list, tuple)) else [ignore_id]
        self.specify_id = None
        if specify_id is not None:
            self.specify_id = specify_id if isinstance(specify_id, (list, tuple)) else [specify_id]

    def __call__(self, result_dict):
        """
        Get binary mask from instance segmentation data.
        required keys: instance_seg
        Update keys:
        Add keys: masks, instance_ids
        """
        instance_seg = result_dict['instance_seg']  # (h, w)
        instance_ids = set(np.unique(instance_seg))
        if self.ignore_id is not None:
            instance_ids -= set(self.ignore_id)
        if self.specify_id is not None:
            instance_ids = set(self.specify_id)

        masks = []
        for l in instance_ids:
            # only one label for one instance
            mask = np.uint8(instance_seg == l)
            masks.append(mask)
        masks = np.stack(masks)  # (n ,h, w)
        result_dict['masks'] = masks
        result_dict['instance_ids'] = np.array(list(instance_ids), np.uint8)
        return result_dict


@TRANSFORM_REGISTRY.registry_module()
class BoxFormMask:
    def __init__(self, bbox_noise_offset=20):
        self.bbox_noise_offset = bbox_noise_offset

    def __call__(self, result_dict):
        """
        Get bounding box from binary mask, the box is the minimum enclosing rectangle of the mask, with a uniform random
        offset to height and width.
        required keys: masks
        Update keys:
        Add keys: boxes
        """
        masks = result_dict['masks']
        max_h, max_w = masks[0].shape

        boxes = []
        for mask in masks:
            y_indices, x_indices = np.where(mask > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            # add perturbation to bounding box coordinates
            H, W = max_h, max_w
            x_min = max(0, x_min - random.randint(0, self.bbox_noise_offset))
            x_max = min(W, x_max + random.randint(0, self.bbox_noise_offset))
            y_min = max(0, y_min - random.randint(0, self.bbox_noise_offset))
            y_max = min(H, y_max + random.randint(0, self.bbox_noise_offset))
            box = np.array([x_min, y_min, x_max, y_max], np.float32)
            boxes.append(box)

        result_dict['boxes'] = np.stack(boxes)
        return result_dict
