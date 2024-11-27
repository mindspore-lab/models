import sys
import os
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from models import build_postprocess


class Postprocessor(object):
    def __init__(self, task="det", algo="DB", **kwargs):
        if task == "e2e":
            if algo.startswith("PG"):
                postproc_cfg = dict(
                    name="PGPostProcess",
                    character_dict_path="models/utils/dict/ic15_dict.txt",
                    score_thresh=0.5,
                    valid_set="totaltext",
                    point_gather_mode="align",
                )
            else:
                raise ValueError(f"No postprocess config defined for {algo}. Please check the algorithm name.")
            self.rescale_internally = True
            self.round = True

        postproc_cfg.update(kwargs)
        self.task = task
        self.postprocess = build_postprocess(postproc_cfg)

    def __call__(self, pred, data=None, **kwargs):
        """
        Args:
            pred: network prediction
            data: (optional)
                preprocessed data, dict, which contains key `shape`
                    - shape: its values are [ori_img_h, ori_img_w, scale_h, scale_w]. scale_h, scale_w are needed to
                      map the predicted polygons back to the orignal image shape.

        return:
            det_res: dict, elements:
                    - polys: shape [num_polys, num_points, 2], point coordinate definition: width (horizontal),
                      height(vertical)
        """
        if self.task == "e2e":
            if self.rescale_internally:
                shape_list = np.array(data["shape_list"], dtype="float32")
                shape_list = np.expand_dims(shape_list, axis=0)
            else:
                shape_list = None
            post_res = self.postprocess(pred, shape_list=shape_list)
            
            return post_res["points"], post_res["texts"]

