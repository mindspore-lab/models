import numpy as np
import cv2

__all__ = ["Resize"]


class Resize:
    def __init__(self, target_size=[320, 640], keep_ratio=True, interp=None):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int, option): the interpolation method
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.interp = interp

    def resize_image(self, image, scale):
        im_scale_x, im_scale_y = scale
        interp = self.interp if self.interp else (cv2.INTER_AREA if min(scale) < 1 else cv2.INTER_LINEAR)

        return cv2.resize(image, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)

    def resize_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)

        return bbox

    def resize_poly(self, poly, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        resized_poly = np.array(poly)
        resized_poly[0::2] = im_scale_x * np.array(poly[0::2])
        resized_poly[0::2] = np.clip(resized_poly[0::2], 0, resize_w)
        resized_poly[1::2] = im_scale_y * np.array(poly[1::2])
        resized_poly[1::2] = np.clip(resized_poly[1::2], 0, resize_h)
        return resized_poly.tolist()

    def __call__(self, img, gt_bbox, gt_class, gt_poly=None):
        """Resize the image numpy."""
        # apply image
        img_shape = img.shape
        if self.keep_ratio:
            im_scale = min(self.target_size[1] / img_shape[1], self.target_size[0] / img_shape[0])
            im_scale_x, im_scale_y = im_scale, im_scale
        else:
            im_scale_y = self.target_size[0] / img_shape[0]
            im_scale_x = self.target_size[1] / img_shape[1]

        img = self.resize_image(img, [im_scale_x, im_scale_y])
        resize_h, resize_w = img.shape[:2]

        if len(gt_bbox) > 0:
            gt_bbox = self.resize_bbox(gt_bbox, [im_scale_x, im_scale_y], [resize_w, resize_h])
        img = cv2.copyMakeBorder(
            img, 0, self.target_size[0] - resize_h, 0, self.target_size[1] - resize_w, cv2.BORDER_CONSTANT
        )  # top, bottom, left, right
        if gt_poly is not None:
            resized_segms = []
            for segm in gt_poly["segmentations"]:
                resized_segms.append(
                    [self.resize_poly(poly, [im_scale_x, im_scale_y], [resize_w, resize_h]) for poly in segm])
            gt_poly["segmentations"] = resized_segms
            return img, gt_bbox, gt_class, gt_poly
        return img, gt_bbox, gt_class
