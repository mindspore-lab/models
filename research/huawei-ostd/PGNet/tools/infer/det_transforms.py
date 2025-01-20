import os
import sys
import cv2

mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
sys.path.insert(0, mindocr_path)

from models.data.transforms import det_transforms

__all__ = ["DetResize", "DetResizeNormForInfer"]


class DetResize(det_transforms.DetResize):
    # limit_type and force_divisable is not supported currently, because inference model don't support dynamic shape
    def __init__(self, keep_ratio=True, padding=True, interpolation=cv2.INTER_LINEAR, **kwargs):
        if keep_ratio and (not padding):
            print(
                "WARNING: output shape can be dynamic if keep_ratio but no padding for DetResize, "
                "but inference doesn't support dynamic shape, so padding is reset to True."
            )
            padding = True

        skipped = ("target_size", "limit_type", "limit_side_len", "force_divisable")
        [kwargs.pop(name, None) for name in skipped]

        super().__init__(
            target_size=None,
            keep_ratio=keep_ratio,
            padding=padding,
            limit_type="none",
            limit_side_len=960,
            force_divisable=False,
            interpolation=interpolation,
            **kwargs,
        )

    def __call__(self, data):
        self.target_size = data["target_size"]
        return super().__call__(data)

