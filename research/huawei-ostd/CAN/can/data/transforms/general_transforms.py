from typing import List, Union

import cv2
import numpy as np
from PIL import Image

from mindspore import dataset as ds
from mindspore.dataset import vision

from ...data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = [
    "DecodeImage",
    "CANImageNormalize",
    "GrayImageChannelFormat",
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
        # data['ori_image'] = img.copy()
        data["raw_img_shape"] = img.shape[:2]

        if self.keep_ori:
            data["image_ori"] = img.copy()
        return data

    
class CANImageNormalize(object):
    """normalize image such as substract mean, divide std"""

    def __init__(self, scale=None, mean=None, std=None, order="chw", **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, data):
        img = data["image"]
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
        data["image"] = (img.astype("float32") * self.scale - self.mean) / self.std
        return data
    
class GrayImageChannelFormat(object):
    """
    format gray scale image's channel: (3,h,w) -> (1,h,w)
    Args:
        inverse: inverse gray image
    """
    def __init__(self, inverse=False, **kwargs):
        self.inverse = inverse

    def __call__(self, data):
        img = data["image"]
        img_single_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_expanded = np.expand_dims(img_single_channel, 0)

        if self.inverse:
            data["image"] = np.abs(img_expanded - 1)
        else:
            data["image"] = img_expanded

        data["src_image"] = img
        return data
