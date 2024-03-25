import mindspore as ms
from mindspore import nn, ops
from mindspore.nn import LossBase

from segment_anything.utils.registry import LOSS_REGISTRY
from segment_anything.utils.utils import calc_iou, reduce_with_mask


def create_loss_fn(args):
    """
    Ref from https://arxiv.org/abs/2304.02643 section3
    SAM loss is the combination of focal and dice loss.
    """
    loss_fn = LOSS_REGISTRY.instantiate(**args)
    return loss_fn


@LOSS_REGISTRY.registry_module()
class SAMLoss(nn.Cell):
    """
    Ref from https://arxiv.org/abs/2304.02643 section3
    SAM loss is the combination of focal and dice loss.
    """
    def __init__(self, focal_factor=20.0, dice_factor=1.0, mse_factor=1.0, mask_threshold=0.0):
        super().__init__()
        self.focal_factor = focal_factor
        self.dice_factor = dice_factor
        self.mse_factor = mse_factor
        self.mask_threshold = mask_threshold

        self.focal_loss = FocalLoss(reduction='none')
        self.dice_loss = DiceLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    def construct(self, pred_mask, pred_iou, gt_mask, valid_boxes):
        """
        get loss, remove dynamic shape assisted with valid_boxes
        Args:
            pred_mask (ms.Tensor): predicted mask with shape (b, n, h, w)
            valid_boxes (ms.Tensor): mask to show whether the input prompt is padded or not. Value 1 means the
            shape (b, n)
        """
        pred_mask = pred_mask.astype(ms.float32)  # (b, n, h, w)  logits, un-simoid
        gt_mask = gt_mask.astype(ms.float32)  # (b, n, h, w)

        pred_mask_01 = (pred_mask > self.mask_threshold).astype(ms.float32)
        gt_iou = ops.stop_gradient(calc_iou(pred_mask_01, gt_mask))  # (b, n)

        # if False: # show pred and gt mask for debug
        #     import matplotlib.pyplot as plt
        #     plt.imshow(pred_mask_01[0, 3].asnumpy())
        #     plt.show()
        #     plt.imshow(gt_mask[0, 3].asnumpy())
        #     plt.show()
        focal_loss = reduce_with_mask(self.focal_loss(pred_mask, gt_mask), valid_boxes)   # (b, n) -> (1,)
        dice_loss = reduce_with_mask(self.dice_loss(pred_mask, gt_mask), valid_boxes)
        mse_loss = reduce_with_mask(self.mse_loss(pred_iou, gt_iou), valid_boxes)
        loss = self.focal_factor * focal_loss + self.dice_factor * dice_loss + self.mse_factor * mse_loss

        return loss, ops.stop_gradient(focal_loss), ops.stop_gradient(dice_loss), ops.stop_gradient(mse_loss)


class DiceLoss(LossBase):
    def __init__(self, reduction='none', smooth=1e-5):
        """Dice loss for 2d segmentation. a replacement for mindspore.nn.DiceLoss that does not
        support reduction 'none' type."""
        super(DiceLoss, self).__init__(reduction)
        self.smooth = smooth

    def construct(self, logits, labels):
        logits = ops.sigmoid(logits)

        shape = logits.shape  # (b, n, h, w)
        # new_shape = (shape[0], shape[1], -1)  # (b, n, s)
        new_shape = (shape[0], shape[1], shape[2]*shape[3])  # (b, n, s)
        logits = logits.view(new_shape)
        labels = labels.view(new_shape)
        intersection = self.reduce_sum(self.mul(logits, labels), -1)  # (b, n, s) -> (b, n)
        unionset = self.reduce_sum(self.mul(logits, logits) + self.mul(labels, labels), -1)

        single_dice_coeff = (2 * intersection) / (unionset + self.smooth)
        dice_loss = 1 - single_dice_coeff

        return dice_loss


class FocalLoss(LossBase):
    def __init__(self, alpha=0.8, gamma=2, reduction='none'):
        """
        Focal loss for 2D binary segmentation. a replacement for mindspore.nn.DiceLoss that only support multi-class.
        FL = - alpha * (1-Pt)**gamma * log(Pt)
        """
        super(FocalLoss, self).__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma

    def construct(self, logits, labels):
        logits = ops.sigmoid(logits)
        shape = logits.shape  # (b, n, h, w)
        # new_shape = (shape[0], shape[1], -1)  # (b, n, s)
        new_shape = (shape[0], shape[1], shape[2]*shape[3])  # (b, n, s)
        logits = logits.view(new_shape)
        labels = labels.view(new_shape)
        bce = ops.binary_cross_entropy(logits, labels, reduction='none').mean(-1)  # (b, n) equals to -log(pt)
        pt = ops.exp(-bce)  # pt
        focal_loss = self.alpha * (1- pt)**self.gamma * bce  # (b, n)
        return focal_loss
