import logging
import math

import cv2
import numpy as np

__all__ = ["DetResize"]
_logger = logging.getLogger(__name__)

class DetResize:
    def __init__(
        self,
        target_size: list = None,
        keep_ratio: bool = True,
        padding: bool = False,
        limit_type: str = "min",
        limit_side_len: int = 736,
        force_divisable: bool = True,
        divisor: int = 32,
        interpolation: int = cv2.INTER_LINEAR,
        **kwargs,
    ):
        if target_size is not None:
            limit_type = "none"

        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.padding = padding
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.interpolation = interpolation
        self.force_divisable = force_divisable
        self.divisor = divisor

        self.is_train = kwargs.get("is_train", False)
        assert target_size is None or limit_type == "none", "Only one of limit_type and target_size should be provided."
        if limit_type in ["min", "max"]:
            keep_ratio = True
            padding = False
            _logger.info(
                f"`limit_type` is {limit_type}. Image will be resized by limiting the {limit_type} "
                f"side length to {limit_side_len}."
            )
        elif limit_type == "none":
            assert target_size is not None or force_divisable is not None, (
                "One of `target_size` or `force_divisable` is required when limit_type is not set. "
                "Please set at least one of them."
            )
            if target_size and force_divisable:
                if (target_size[0] % divisor != 0) or (target_size[1] % divisor != 0):
                    self.target_size = [max(round(x / self.divisor) * self.divisor, self.divisor) for x in target_size]
                    _logger.warning(
                        f"`force_divisable` is enabled but the set target size {target_size} "
                        f"is not divisable by {divisor}. Target size is ajusted to {self.target_size}"
                    )
            if (target_size is not None) and keep_ratio and (not padding):
                _logger.warning("output shape can be dynamic if keep_ratio but no padding.")
        else:
            raise ValueError(f"Unknown limit_type: {limit_type}")

    def __call__(self, data: dict) -> dict:
        img = data["image"]
        h, w = img.shape[:2]
        if self.target_size:
            tar_h, tar_w = self.target_size

        scale_ratio = 1.0
        allow_padding = False
        if self.limit_type == "min":
            if min(h, w) < self.limit_side_len:
                scale_ratio = self.limit_side_len / float(min(h, w))
        elif self.limit_type == "max":
            if max(h, w) > self.limit_side_len:
                scale_ratio = self.limit_side_len / float(max(h, w))
        elif self.limit_type == "none":
            if self.keep_ratio and self.target_size:
                scale_ratio = min(tar_h / h, tar_w / w)
                allow_padding = True

        if (self.limit_type in ["min", "max"]) or (self.target_size and self.keep_ratio):
            resize_h = int(h * scale_ratio)
            resize_w = int(w * scale_ratio)
            resize_h = max(int(round(resize_h / 32) * 32), 32)
            resize_w = max(int(round(resize_w / 32) * 32), 32)
            if self.target_size:
                resize_w = min(resize_w, tar_w)
                resize_h = min(resize_h, tar_h)
        elif self.target_size:
            resize_w = tar_w
            resize_h = tar_h
        else:
            resize_w = w
            resize_h = h

        if self.force_divisable:
            if not (
                allow_padding and self.padding
            ):  
                resize_h = max(
                    math.ceil(resize_h / self.divisor) * self.divisor, self.divisor
                )
                resize_w = max(math.ceil(resize_w / self.divisor) * self.divisor, self.divisor)

        resized_img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interpolation)

        if allow_padding and self.padding:
            if self.target_size and (tar_h >= resize_h and tar_w >= resize_w):
                padded_img = np.zeros((tar_h, tar_w, 3), dtype=np.uint8)
                padded_img[:resize_h, :resize_w, :] = resized_img
                data["image"] = padded_img
            else:
                _logger.warning(
                    f"Image shape after resize is ({resize_h}, {resize_w}), "
                    f"which is larger than target_size {self.target_size}. Skip padding for the current image. "
                    f"You may disable `force_divisable` to avoid this warning."
                )
        else:
            data["image"] = resized_img

        scale_h = resize_h / h
        scale_w = resize_w / w

        if "polys" in data and len(data["polys"]) and self.is_train:
            data["polys"][:, :, 0] = data["polys"][:, :, 0] * scale_w
            data["polys"][:, :, 1] = data["polys"][:, :, 1] * scale_h

        if "shape_list" not in data:
            src_h, src_w = data.get("raw_img_shape", (h, w))
            data["shape_list"] = np.array([src_h, src_w, scale_h, scale_w], dtype=np.float32)
        else:
            data["shape_list"][2] = data["shape_list"][2] * scale_h
            data["shape_list"][3] = data["shape_list"][3] * scale_h

        return data
