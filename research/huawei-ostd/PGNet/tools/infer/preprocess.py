import logging
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from models.data.transforms import create_transforms, run_transforms

_logger = logging.getLogger("mindocr")

from models.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class Preprocessor(object):
    def __init__(self, task="det", algo="DB", **kwargs):
        if task == "det":
            limit_side_len = kwargs.get("det_limit_side_len", 736)
            limit_type = kwargs.get("det_limit_type", "min")

            pipeline = [
                {"DecodeImage": {"img_mode": "RGB", "keep_ori": True, "to_float32": False}},
                {
                    "DetResize": {
                        "target_size": None,
                        "keep_ratio": True,
                        "limit_side_len": limit_side_len,
                        "limit_type": limit_type,
                        "padding": False,
                        "force_divisable": True,
                    }
                },
                {
                    "NormalizeImage": {
                        "bgr_to_rgb": False,
                        "is_hwc": True,
                        "mean": IMAGENET_DEFAULT_MEAN,
                        "std": IMAGENET_DEFAULT_STD,
                    }
                },
                {"ToCHWImage": None},
            ]
            _logger.info(f"Pick optimal preprocess hyper-params for det algo {algo}:\n {pipeline[1]}")

        self.pipeline = pipeline
        self.transforms = create_transforms(pipeline)

    def __call__(self, img_or_path):
        """
        Return:
            dict, preprocessed data containing keys:
                - image: np.array, transfomred image
                - image_ori: np.array, original image
                - shape: list of [ori_h, ori_w, scale_h, scale_w]
                and other keys added in transform pipeline.
        """
        if isinstance(img_or_path, str):
            data = {"img_path": img_or_path}
            output = run_transforms(data, self.transforms)
        elif isinstance(img_or_path, dict):
            output = run_transforms(img_or_path, self.transforms)
        else:
            data = {"image": img_or_path}
            data["image_ori"] = img_or_path.copy()  # TODO
            data["image_shape"] = img_or_path.shape
            output = run_transforms(data, self.transforms[1:])

        return output
