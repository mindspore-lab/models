import glob
import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

_logger = logging.getLogger("mindocr")


def get_image_paths(img_dir: str) -> List[str]:
    """
    Args:
        img_dir: path to an image or a directory containing multiple images.

    Returns:
        List: list of image paths in the directory and its subdirectories.
    """
    img_dir = Path(img_dir)
    assert img_dir.exists(), f"{img_dir} does NOT exist. Please check the directory / file path."

    extensions = [".jpg", ".png", ".jpeg"]
    if img_dir.is_file():
        img_paths = [str(img_dir)]
    else:
        img_paths = [str(file) for file in img_dir.rglob("*.*") if file.suffix.lower() in extensions]

    assert (
        len(img_paths) > 0
    ), f"{img_dir} does NOT exist, or no image files exist in {img_dir}. Please check the `image_dir` arg value."
    return sorted(img_paths)


def get_ckpt_file(ckpt_dir):
    if os.path.isfile(ckpt_dir):
        ckpt_load_path = ckpt_dir
    else:
        ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
        assert len(ckpt_paths) != 0, f"No .ckpt files found in {ckpt_dir}"
        ckpt_load_path = ckpt_paths[0]
        if len(ckpt_paths) > 1:
            _logger.warning(f"More than one .ckpt files found in {ckpt_dir}. Pick {ckpt_load_path}")

    return ckpt_load_path

def draw_e2e_res(dt_boxes, strs, img_path):
    src_im = cv2.imread(img_path)
    for box, str in zip(dt_boxes, strs):
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        cv2.putText(
            src_im,
            str,
            org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=1,
        )
    return src_im
