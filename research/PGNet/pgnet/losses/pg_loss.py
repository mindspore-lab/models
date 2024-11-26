import numpy as np

import mindspore as ms
from mindspore import nn, Tensor
import mindspore.ops as ops
from pgnet.utils.extract_batchsize import pre_process

__all__ = ["PGLoss"]

class DiceLoss(nn.LossBase):
    """
    Introduced in `"Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations"
    <https://arxiv.org/abs/1905.02244>`_. Dice loss handles well the class imbalance in terms of pixel count for
    foreground and background.

    Args:
        eps: epsilon value to add to the denominator to avoid division by zero. Default: 1e-6.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._eps = eps

    def construct(self, pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
        pred = pred.squeeze(axis=1) * mask
        gt = gt.squeeze(axis=1) * mask

        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() + self._eps
        return 1 - 2.0 * intersection / union

class PGLoss(nn.LossBase):
    def __init__(
        self, tcl_bs, max_text_length, max_text_nums, pad_num, eps=1e-6, **kwargs
    ):
        super(PGLoss, self).__init__()
        self.tcl_bs = tcl_bs
        self.max_text_nums = max_text_nums
        self.max_text_length = max_text_length
        self.pad_num = pad_num
        self.dice_loss = DiceLoss(eps=eps)
        self.cast = ops.Cast()

    def border_loss(self, f_border, l_border, l_score, l_mask):
        l_border_split, l_border_norm = ops.split(
            l_border, (4, 1), axis=1
        )
        f_border_split = f_border
        b, c, h, w = ops.shape(l_border_norm)
        l_border_norm_split = ops.broadcast_to(l_border_norm, (b, 4 * c, h, w))
        b, c, h, w = ops.shape(l_score)
        l_border_score = ops.broadcast_to(l_score, (b, 4 * c, h, w))
        b, c, h, w = ops.shape(l_mask)
        l_border_mask = ops.broadcast_to(l_mask, (b, 4 * c, h, w))
        border_diff = l_border_split - f_border_split
        abs_border_diff = ops.abs(border_diff)
        border_sign = abs_border_diff < 1.0
        border_sign = self.cast(border_sign, ms.float32)
        border_sign.stop_gradient = True
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + (
            abs_border_diff - 0.5
        ) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = ops.sum(border_out_loss * l_border_score * l_border_mask) / (
            ops.sum(l_border_score * l_border_mask) + 1e-5
        )
        return border_loss

    def direction_loss(self, f_direction, l_direction, l_score, l_mask):
        l_direction_split, l_direction_norm = ops.split(
            l_direction, (2, 1), axis=1
        )
        f_direction_split = f_direction
        b, c, h, w = l_direction_norm.shape
        l_direction_norm_split = ops.broadcast_to(
            l_direction_norm, (b, 2 * c, h, w)
        )
        b, c, h, w = ops.shape(l_score)
        l_direction_score = ops.broadcast_to(l_score, (b, 2 * c, h, w))
        b, c, h, w = ops.shape(l_mask)
        l_direction_mask = ops.broadcast_to(l_mask, (b, 2 * c, h, w))
        direction_diff = l_direction_split - f_direction_split
        abs_direction_diff = ops.abs(direction_diff)
        direction_sign = abs_direction_diff < 1.0
        direction_sign = self.cast(direction_sign, ms.int32)
        direction_sign.stop_gradient = True
        direction_in_loss = (
            0.5 * abs_direction_diff * abs_direction_diff * direction_sign
            + (abs_direction_diff - 0.5) * (1.0 - direction_sign)
        )
        direction_out_loss = l_direction_norm_split * direction_in_loss
        direction_loss = ops.sum(
            direction_out_loss * l_direction_score * l_direction_mask
        ) / (ops.sum(l_direction_score * l_direction_mask) + 1e-5)
        return direction_loss

    def ctcloss(self, f_char, tcl_pos, tcl_mask, tcl_label, label_t):
        f_char = ops.transpose(f_char, (0, 2, 3, 1))
        tcl_pos = ops.reshape(tcl_pos, (-1, 3))
        tcl_pos = self.cast(tcl_pos, ms.int64)
        f_tcl_char = ops.gather_nd(f_char, tcl_pos)
        f_tcl_char = ops.reshape(
            f_tcl_char, (-1, 64, self.pad_num + 1)
        )

        f_tcl_char_fg, f_tcl_char_bg = ops.split(
            f_tcl_char, (self.pad_num, 1), axis=2
        )
        f_tcl_char_bg = f_tcl_char_bg * tcl_mask + (1.0 - tcl_mask) * 20.0

        b, c, l = ops.shape(tcl_mask)
        tcl_mask_fg = ops.broadcast_to(tcl_mask, (b, c, self.pad_num * l))
        tcl_mask_fg.stop_gradient = True

        f_tcl_char_fg = f_tcl_char_fg * tcl_mask_fg + (1.0 - tcl_mask_fg) * (-20.0)
        f_tcl_char_mask = ops.cat((f_tcl_char_fg, f_tcl_char_bg), axis=2)
        f_tcl_char_ld = ops.transpose(f_tcl_char_mask, (1, 0, 2))
        N, B, _ = f_tcl_char_ld.shape
        input_lengths = ms.Tensor(np.array([N] * B, dtype=np.int64), ms.int64)
        log_softmax = nn.LogSoftmax()
        f_tcl_char_ld = log_softmax(f_tcl_char_ld)
        ctc_loss = nn.CTCLoss(blank=self.pad_num, reduction="none", zero_infinity=False)
        cost= ctc_loss(f_tcl_char_ld, tcl_label, input_lengths, label_t)
        mean_cost = ops.reduce_mean(cost)
        return mean_cost
    
    def construct(
        self, predicts, tcl_maps, tcl_label_maps, border_maps, direction_maps, training_masks, label_list, pos_list, pos_mask
        ):
        pos_list, pos_mask, label_list, label_t = pre_process(
            label_list,
            pos_list,
            pos_mask,
            self.max_text_length,
            self.max_text_nums,
            self.pad_num,
            self.tcl_bs,
        )
        f_score, f_border, f_direction, f_char = (
            predicts["f_score"],
            predicts["f_border"],
            predicts["f_direction"],
            predicts["f_char"],
        )
        score_loss = self.dice_loss(f_score, tcl_maps, training_masks)
        border_loss = self.border_loss(f_border, border_maps, tcl_maps, training_masks)
        direction_loss = self.direction_loss(
            f_direction, direction_maps, tcl_maps, training_masks
        )
        ctc_loss = self.ctcloss(f_char, pos_list, pos_mask, label_list, label_t)
        loss_all = score_loss + border_loss + direction_loss + 5 * ctc_loss
        return loss_all


