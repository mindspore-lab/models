import random
from typing import Tuple, Dict, List

import numpy as np

from segment_anything.utils.registry import TRANSFORM_REGISTRY
from segment_anything.utils.transforms import ResizeLongestSide, resize_no_alias


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
            from segment_anything.utils.visualize import show_box, show_mask
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

        Required keys: masks, boxes, image_patches(Option)
        update keys: masks, boxes, image_patches(Option)
        add keys: valid_boxes
        """

        boxes = result_dict['boxes']
        masks = result_dict['masks']
        image_patches = result_dict.get('image_patches')

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

        if image_patches is not None:
            # (nv, c, hp, wp) normed with Blip2ImageProcess
            nv, c, hp, wp = image_patches.shape
            pad_len = 0 if valid_gt_size > self.gt_size else self.gt_size - valid_gt_size
            image_patches = image_patches[:self.gt_size]  # keep at most gt_size samples
            image_patches = np.concatenate([image_patches, np.zeros((pad_len, c, hp, wp), np.float32)], axis=0)
            result_dict['image_patches'] = image_patches  # (n, c, h, w)

        result_dict['boxes'] = boxes  # (n, 4)
        result_dict['masks'] = masks  # (n, h, w)
        result_dict['valid_boxes'] = np.array(valid_boxes, dtype=np.uint8)
        return result_dict


@TRANSFORM_REGISTRY.registry_module()
class ImageResizeAndPad:

    def __init__(self, target_size, apply_box=True, apply_mask=True, apply_point=False):
        """
        Args:
            target_size (int): target size of model input (1024 in sam)
            apply_box: also resize box accordingly beside image
            apply_mask: also resize and pad mask accordingly beside image
            apply_mask: also resize point accordingly beside image
        """
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.apply_box = apply_box
        self.apply_mask = apply_mask
        self.apply_point = apply_point

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
        masks = result_dict.get('masks')
        boxes = result_dict.get('boxes')
        points = result_dict.get('points')

        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        resized_h, resized_w, _ = image.shape

        # Pad image and masks to the model input
        h, w, c = image.shape
        max_dim = max(h, w)  # long side length
        assert max_dim == self.target_size
        # pad 0 to the right and bottom side
        pad_h = max_dim - h
        pad_w = max_dim - w
        img_padding = ((0, pad_h), (0, pad_w), (0, 0))
        image = np.pad(image, pad_width=img_padding, constant_values=0)  # (h, w, c)

        result_dict['origin_hw'] = np.array([og_h, og_w, resized_h, resized_w], np.int32)  # record image shape trace for visualization
        result_dict['image'] = image
        result_dict['image_pad_area'] = img_padding[:2]

        if self.apply_box:
            # Adjust bounding boxes
            boxes = self.transform.apply_boxes(boxes, (og_h, og_w)).astype(np.float32)
            result_dict['boxes'] = boxes

        if self.apply_mask:
            masks = np.stack([self.transform.apply_image(mask) for mask in masks])
            mask_padding = ((0, 0), (0, pad_h), (0, pad_w))  # (n, h, w)
            masks = np.pad(masks, pad_width=mask_padding, constant_values=0)
            result_dict['masks'] = masks

        if self.apply_point:
            points = self.transform.apply_coords(points, (og_h, og_w)).astype(np.float32)
            result_dict['points'] = points

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
        if False:  # show image and mask for debug
            import matplotlib.pyplot as plt
            plt.imshow(result_dict['image'].transpose([1,2,0]))  # raw image
            from segment_anything.utils.visualize import show_box, show_mask
            show_box(result_dict['boxes'][0], plt.gca())
            show_mask(result_dict['masks'][0], plt.gca())
            plt.show()

        return result_dict


@TRANSFORM_REGISTRY.registry_module()
class ImagePatchFromBoxMask:
    """
    This is the approach to generate text training prompts described in D.5. Zero-Shot Text-to-Mask of the officail SAM paper

    """
    def __init__(self, random_factor_range=(1.0, 2.0), resize=336):
        self.random_factor_range = random_factor_range
        self.resize = resize

    def __call__(self, result_dict):
        """
        Get image patch from raw image with box and mask by following steps
            1. Expand the bounding box by a random factor.
            2. Square-crop the expanded box to maintain its aspect ratio
            3. crop the image and resize it to 336*336 pixel
            4. Zero-out the non-mask area with 50% probability
        required keys: masks, boxes, image
        Update keys: None
        Add keys: image_patches
        """
        full_image = result_dict['image']  # raw image (h, w, c)
        masks = result_dict['masks']  # (n, h, w) [0, 1]
        boxes = result_dict['boxes']  # (n, 4)

        max_h, max_w = masks[0].shape

        image_patches = []
        patch_boxes = []
        for box, mask in zip(boxes, masks):
            x1, y1, x2, y2 = box
            cx, cy, w, h = (x2+x1)/2, (y2+y1)/2, (x2-x1), (y2-y1)
            factor = np.random.uniform(low=self.random_factor_range[0], high=self.random_factor_range[1])
            exp_x1 = max(int(cx - factor*w/2), 0)
            exp_y1 = max(int(cy - factor*h/2), 0)
            exp_x2 = min(int(cx + factor*w/2), max_w)
            exp_y2 = min(int(cy + factor*h/2), max_h)

            exp_cx = (exp_x1+exp_x2)/2
            exp_cy = (exp_y1+exp_y2)/2
            exp_w = exp_x2 - exp_x1
            exp_h = exp_y2 - exp_y1

            side_length = min(exp_w, exp_h)
            square_x1 = max(int(exp_cx - side_length/2), 0)
            square_y1 = max(int(exp_cy - side_length/2), 0)
            square_x2 = min(int(exp_cx + side_length/2), max_w)
            square_y2 = min(int(exp_cy + side_length/2), max_h)

            assert full_image.shape[2] == 3
            image_patch = full_image[square_y1: square_y2, square_x1: square_x2]

            image_patch = resize_no_alias(image_patch, (self.resize, self.resize))

            if np.random.uniform() > 0.5:
                mask_patch = mask[square_y1: square_y2, square_x1: square_x2]
                # value int[0, 1], (n, h, w)
                mask_patch = resize_no_alias(mask_patch, (self.resize, self.resize))
                image_patch = image_patch * np.expand_dims(mask_patch, -1)  # (h, w, c)

            image_patches.append(image_patch)
            patch_boxes.append([square_x1, square_y1, square_x2, square_y2])

        result_dict['image_patches'] = np.stack(image_patches)  # (n, h, w, c)

        if False:  # show image and mask for debug
            imd = 0
            import matplotlib.pyplot as plt
            plt.imshow(result_dict['image'])  # raw image
            from segment_anything.utils.visualize import show_box, show_mask
            show_box(patch_boxes[imd], plt.gca())
            show_mask(result_dict['masks'][imd], plt.gca())
            plt.show()
            plt.imshow(result_dict['image_patches'][imd])  # normed image
            plt.show()

        return result_dict


@TRANSFORM_REGISTRY.registry_module()
class ImagePatchPreprocess:
    """
    This is the approach to generate text training prompts described in D.5. Zero-Shot Text-to-Mask of the officail SAM paper

    """
    def __init__(self, model='blip2_stage1_classification'):
        from mindformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model)

    def __call__(self, result_dict):
        """
        Process image patch to blip2 model input
        required keys: image_patches
        Update keys: image_patches
        Add keys: None
        """
        image_patches = result_dict["image_patches"]  # (n, c, h, w), in range(0, 255), RGB
        processed_patches = self.processor.image_processor(image_patches)  # (n, c, 224, 224), in range(0, 1), RGB
        result_dict["image_patches"] = processed_patches.asnumpy()

        return result_dict
