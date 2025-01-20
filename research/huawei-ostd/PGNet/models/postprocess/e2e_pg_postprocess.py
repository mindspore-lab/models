import mindspore as ms
from models.utils.extract_textpoint import get_dict, generate_pivot_list_fast, restore_poly

__all__ = ["PGPostProcess"]


class PGPostProcess(object):
    def __init__(
        self,
        character_dict_path,
        valid_set,
        score_thresh,
        point_gather_mode=None,
    ):
        self.Lexicon_Table = get_dict(character_dict_path)
        self.valid_set = valid_set
        self.score_thresh = score_thresh
        self.point_gather_mode = point_gather_mode

    def __call__(self, outs_dict, shape_list, **kwargs):
        p_score = outs_dict["f_score"]
        p_border = outs_dict["f_border"]
        p_char = outs_dict["f_char"]
        p_direction = outs_dict["f_direction"]
        if isinstance(p_score, ms.Tensor):
            p_score = p_score[0].numpy()
            p_border = p_border[0].numpy()
            p_direction = p_direction[0].numpy()
            p_char = p_char[0].numpy()
        else:
            p_score = p_score[0]
            p_border = p_border[0]
            p_direction = p_direction[0]
            p_char = p_char[0]

        src_h, src_w, ratio_h, ratio_w = shape_list[0]
        instance_yxs_list, seq_strs = generate_pivot_list_fast(
            p_score,
            p_char,
            p_direction,
            self.Lexicon_Table,
            score_thresh=self.score_thresh,
            point_gather_mode=self.point_gather_mode,
        )
        poly_list, keep_str_list = restore_poly(
            instance_yxs_list,
            seq_strs,
            p_border,
            ratio_w,
            ratio_h,
            src_w,
            src_h,
            self.valid_set,
        )
        data = {
            "points": poly_list,
            "texts": keep_str_list,
        }
        return data
