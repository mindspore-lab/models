import mindspore as ms
from mindspore import ops, nn
import numpy as np


class AllReduce(nn.Cell):
    def __init__(self):
        super(AllReduce, self).__init__()
        self.all_reduce_sum = ops.AllReduce(ops.ReduceOp.SUM)

    def construct(self, x):
        return self.all_reduce_sum(x)


class GetConfusionMatrix:
    """
    Calcute the confusion matrix by given label and pred.
    """
    def __init__(self, num_class, ignore=-1, rank_size=1):
        super(GetConfusionMatrix, self).__init__()
        self.all_reduce_sum = AllReduce
        self.num_class = num_class
        self.ignore = ignore
        self.rank_size = rank_size

    def __call__(self, label, pred):
        seg_gt = label.astype(dtype=np.int32)

        ignore_index = seg_gt != self.ignore
        seg_gt = seg_gt[ignore_index]
        seg_pred = pred[ignore_index]

        index = (seg_gt * self.num_class + seg_pred).astype("int32")
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((self.num_class, self.num_class))

        for i_label in range(self.num_class):
            for i_pred in range(self.num_class):
                cur_index = i_label * self.num_class + i_pred
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred] = label_count[cur_index]
        if self.rank_size > 1:
            confusion_matrix = ms.Tensor(confusion_matrix, ms.float32)
            confusion_matrix = self.all_reduce_sum(confusion_matrix).asnumpy()
        return confusion_matrix
