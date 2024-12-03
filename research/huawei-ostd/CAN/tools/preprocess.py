import logging
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from can.data.transforms import create_transforms, run_transforms

_logger = logging.getLogger("mindocr")


class Preprocessor(object):
    def __init__(self, task="rec", algo="CAN", **kwargs):
        if task == "rec" and algo == "CAN":
            pipeline = [
                {"DecodeImage": {
                    "img_mode": "BGR",
                    "channel_first": False,
                    },
                },
                {"CANImageNormalize": {
                    "mean": [0,0,0],
                    "std": [1,1,1],
                    "order": 'hwc',
                    },
                },
                {"GrayImageChannelFormat": {
                    "inverse": True
                    },
                },
                {"CANLabelEncode":{
                    "task": "infer"
                    },
                },
            ]
        else:
            raise ValueError(f"This project is only for the REC_CAN model.")

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
