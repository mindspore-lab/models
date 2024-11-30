import numpy as np

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.nn.loss.loss import LossBase

__all__ = ["CANLoss"]


class CANLoss(LossBase):
    '''
    CANLoss is consist of two part:
        word_average_loss: average accuracy of the symbol
        counting_loss: counting loss of every symbol
    '''

    def __init__(self):
        super().__init__()

        self.use_label_mask = False
        self.out_channel = 111
        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')
        self.ratio = 16

    def construct(self, preds, labels_in, label_masks_in):
        word_probs = preds["word_probs"]
        counting_preds = preds["counting_preds"]
        counting_preds1 = preds["counting_preds1"]
        counting_preds2 = preds["counting_preds2"]

        labels = labels_in
        labels_mask = label_masks_in
        counting_labels = self.gen_counting_label(labels, self.out_channel)
        counting_loss = (self.counting_loss(counting_preds1, counting_labels)
                         + self.counting_loss(counting_preds2, counting_labels)
                         + self.counting_loss(counting_preds, counting_labels))
        word_loss = self.cross(word_probs.view(-1, word_probs.shape[-1]), labels.view(-1))
        word_average_loss = (word_loss * labels_mask.view(-1)).sum() / (
            labels_mask.sum() + 1e-10
        ) if self.use_label_mask else word_loss
        loss = word_average_loss + counting_loss
        return loss

    def gen_counting_label(self, labels, channel, tag=True):
        b, t = labels.shape
        counting_labels = np.zeros([b, channel])

        if tag:
            ignore = [0, 1, 107, 108, 109, 110]
        else:
            ignore = []
        for i in range(b):
            for j in range(t):
                k = labels[i][j]
                if k in ignore:
                    continue
                else:
                    counting_labels[i][k] += 1
        counting_labels = Tensor(counting_labels, dtype=ms.float32)
        return counting_labels
