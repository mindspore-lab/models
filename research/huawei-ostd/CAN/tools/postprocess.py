import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from can import build_postprocess


class Postprocessor(object):
    def __init__(self, task="rec", algo="CAN", rec_char_dict_path=None, **kwargs):
        if task == "rec" and algo.startswith("CAN"):
            postproc_cfg = dict(
                name="CANLabelDecode",
                character_dict_path=rec_char_dict_path,
                use_space_char=False,
            )
        else:
            raise ValueError(f"This project is only for the REC_CAN model.")

        postproc_cfg.update(kwargs)
        self.task = task
        self.postprocess = build_postprocess(postproc_cfg)

    def __call__(self, pred, data=None, **kwargs):
        output = self.postprocess(pred)
        return output
