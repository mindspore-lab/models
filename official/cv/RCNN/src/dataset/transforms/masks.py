import cv2
import numpy as np
from pycocotools import mask as maskHelper

__all__ = ["Poly2Mask"]


class Poly2Mask:
    """
    Poly style to mask
    Args:
        padding_size (int): samples target padding to this size, if targets size greater than padding_size, crop to this size.
    """
    def ann_to_mask(self, segm, height, width):
        """Convert annotation to RLE and then to binary mask."""
        if isinstance(segm, list):
            rles = maskHelper.frPyObjects(segm, height, width)
            rle = maskHelper.merge(rles)
        elif isinstance(segm["counts"], list):
            rle = maskHelper.frPyObjects(segm, height, width)
        else:
            rle = segm
        m = maskHelper.decode(rle)
        return m

    def __call__(self, img, gt_bbox, gt_class, gt_poly=None):
        """
        Padding the list of numpy labels.
        """
        if gt_poly is not None:
            h, w = img.shape[1:3]
            num_bbox = gt_bbox.shape[0]
            gt_mask =np.zeros((num_bbox, h, w), dtype=np.bool_)
            for i, segs in enumerate(gt_poly["segmentations"]):
                if len(segs) == 0:
                    gt_mask[i] = np.zeros((h, w), np.bool_)
                else:
                    gt_mask[i] = self.ann_to_mask(segs, h, w).astype(np.bool_)
            return img, gt_bbox, gt_class, gt_mask
        return img, gt_bbox, gt_class
