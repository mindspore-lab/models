from typing import List, Union

import cv2
import numpy as np
from PIL import Image

from mindspore.dataset import vision

IMAGENET_DEFAULT_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_DEFAULT_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]

__all__ = [
    "DecodeImage",
    "NormalizeImage",
    "ToCHWImage",
]

def get_value(val, name):
    if isinstance(val, str) and val.lower() == "imagenet":
        assert name in ["mean", "std"]
        return IMAGENET_DEFAULT_MEAN if name == "mean" else IMAGENET_DEFAULT_STD
    elif isinstance(val, list):
        return val
    else:
        raise ValueError(f"Wrong {name} value: {val}")

class DecodeImage:
    """
    img_mode (str): The channel order of the output, 'BGR' and 'RGB'. Default to 'BGR'.
    channel_first (bool): if True, image shpae is CHW. If False, HWC. Default to False
    """
    def __init__(
        self, img_mode="BGR", channel_first=False, to_float32=False, ignore_orientation=False, keep_ori=False, **kwargs
    ):
        self.img_mode = img_mode
        self.to_float32 = to_float32
        self.channel_first = channel_first
        self.flag = cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR if ignore_orientation else cv2.IMREAD_COLOR
        self.keep_ori = keep_ori

        self.use_minddata = kwargs.get("use_minddata", False)
        self.decoder = None
        self.cvt_color = None
        if self.use_minddata:
            self.decoder = vision.Decode()
            self.cvt_color = vision.ConvertColor(vision.ConvertMode.COLOR_BGR2RGB)

    def __call__(self, data):
        if "img_path" in data:
            with open(data["img_path"], "rb") as f:
                img = f.read()
        elif "img_lmdb" in data:
            img = data["img_lmdb"]
        else:
            raise ValueError('"img_path" or "img_lmdb" must be in input data')
        img = np.frombuffer(img, dtype="uint8")

        if self.use_minddata:
            img = self.decoder(img)
            if self.img_mode == "BGR":
                img = self.cvt_color(img)
        else:
            img = cv2.imdecode(img, self.flag)
            if self.img_mode == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        if self.to_float32:
            img = img.astype("float32")
        data["image"] = img
        data["raw_img_shape"] = img.shape[:2]

        if self.keep_ori:
            data["image_ori"] = img.copy()
        return data


class NormalizeImage:
    """
    normalize image, subtract mean, divide std
    input image: by default, np.uint8, [0, 255], HWC format.
    return image: float32 numpy array
    """
    def __init__(
        self,
        mean: Union[List[float], str] = "imagenet",
        std: Union[List[float], str] = "imagenet",
        is_hwc=True,
        bgr_to_rgb=False,
        rgb_to_bgr=False,
        **kwargs,
    ):
        # By default, imagnet MEAN and STD is in RGB order. inverse if input image is in BGR mode
        self._channel_conversion = False
        if bgr_to_rgb or rgb_to_bgr:
            self._channel_conversion = True

        shape = (3, 1, 1) if not is_hwc else (1, 1, 3)
        self.mean = get_value(mean, "mean")
        self.std = get_value(std, "std")
        self.is_hwc = is_hwc

        self.use_minddata = kwargs.get("use_minddata", False)
        self.normalize = None
        self.cvt_color = None
        if self.use_minddata:
            self.decoder = vision.Normalize(self.mean, self.std, is_hwc)
            self.cvt_color = vision.ConvertColor(vision.ConvertMode.COLOR_BGR2RGB)
        else:
            self.mean = np.array(self.mean).reshape(shape).astype("float32")
            self.std = np.array(self.std).reshape(shape).astype("float32")

    def __call__(self, data):
        img = data["image"]
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"

        if self.use_minddata:
            if self._channel_conversion:
                img = self.cvt_color(img)
            img = self.normalize(img)
            data["image"] = img
            return data

        if self._channel_conversion:
            if self.is_hwc:
                img = img[..., [2, 1, 0]]
            else:
                img = img[[2, 1, 0], ...]

        data["image"] = (img.astype("float32") - self.mean) / self.std
        return data


class ToCHWImage:
    # convert hwc image to chw image
    def __init__(self, **kwargs):
        self.use_minddata = kwargs.get("use_minddata", False)
        self.hwc2chw = None
        if self.use_minddata:
            self.hwc2chw = vision.HWC2CHW()

    def __call__(self, data):
        img = data["image"]
        if isinstance(img, Image.Image):
            img = np.array(img)
        if self.use_minddata:
            data["image"] = self.hwc2chw(img)
            return data
        data["image"] = img.transpose((2, 0, 1))
        return data
