import mindspore
import mindspore.nn as nn


class dice_bce_loss(nn.LossBase):
    def __init__(self, batch=True, reduction="mean"):
        super(dice_bce_loss, self).__init__(reduction)
        self.batch = batch
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.sum = mindspore.ops.ReduceSum(keep_dims=False)

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = self.sum(y_true)
            j = self.sum(y_pred)
            intersection = self.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def construct(self, predict, target):
        a = self.bce_loss(predict, target)
        b = self.soft_dice_loss(target, predict)
        return a + b


class iou_bce_loss(nn.LossBase):
    def __init__(self, batch=True, reduction="mean"):
        super(iou_bce_loss, self).__init__(reduction)
        self.batch = batch
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.sum = mindspore.ops.ReduceSum(keep_dims=False)

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = self.sum(y_true)
            j = self.sum(y_pred)
            intersection = self.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (intersection + smooth) / (i + j - intersection + smooth)  # iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def construct(self, predict, target):
        a = self.bce_loss(predict, target)
        b = self.soft_dice_loss(target, predict)
        return a + b


def create_loss(
    name: str = "bce",
):
    r"""Creates loss function

    Args:
        name (str):  loss name : 'bce': binary cross entropy. Default: 'CE'.

    Inputs:
        - logits (Tensor or Tuple of Tensor): Input logits. Shape [N, C], where N means the number of samples,
            C means number of classes. Tuple of two input logits are supported in order (main_logits, aux_logits)
            for auxiliary loss used in networks like inception_v3. Data type must be float16 or float32.
        - labels (Tensor): Ground truth labels. Shape: [N] or [N, C].
            (1) If in shape [N], sparse labels representing the class indices. Must be int type.
            (2) shape [N, C], dense labels representing the ground truth class probability values,
            or the one-hot labels. Must be float type. If the loss type is BCE, the shape of labels must be [N, C].

    Returns:
       Loss function to compute the loss between the input logits and labels.
    """
    name = name.lower()

    if name == "bce":
        loss = dice_bce_loss()
    else:
        raise NotImplementedError

    return loss
