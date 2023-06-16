import mindspore as ms
from mindspore import ops, nn


class CrossEntropy(nn.LossBase):
    """
    Cross-entropy loss function for semantic segmentation,
    and different classes have different weights.

    Args:
        num_classes (int): The rescaling weight to each class.
        ignore_label (int): Specifies a target value that is ignored.
        cls_weight (list):  The rescaling weight to each class.
    """

    def __init__(self, num_classes=19, ignore_label=255, cls_weight=None):
        super(CrossEntropy, self).__init__()
        self.num_classes = num_classes
        weight = ms.Tensor([1.0] * num_classes, ms.float32)
        self.ignore_label = ignore_label
        if cls_weight is not None:
            weight = ms.Tensor(cls_weight, ms.float32)
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction="mean")
        # self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")

    def construct(self, logits, labels):
        """Loss construction."""
        labels = labels.astype(ms.int32).reshape((-1,))
        labels = ops.select(labels >= 0, labels, ops.ones_like(labels) * self.ignore_label)
        logits = ops.transpose(logits, (0, 2, 3, 1)).reshape((-1, self.num_classes))
        loss = self.ce(logits, labels)
        # loss = self.ce(logits, ops.select(labels != self.ignore_label, labels, ops.zeros_like(labels)))
        # loss_mask = (labels != self.ignore_label).astype(loss.dtype).reshape(loss.shape)
        # loss = (loss_mask * loss).sum() / (loss_mask.sum() + 1e-6)
        return loss
