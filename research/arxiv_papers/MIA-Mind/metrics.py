import numpy as np
import mindspore.ops as ops

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------- U-Net 风格指标 --------------------
class UnetMetrics:
    def __init__(self, threshold=0.5):
        self.sigmoid = ops.Sigmoid()
        self.threshold = threshold

    def compute(self, logits, targets):
        probs = self.sigmoid(logits)
        preds = probs > self.threshold

        preds = preds.asnumpy().astype(bool)
        targets = targets.asnumpy().astype(bool)

        inter = np.logical_and(preds, targets).sum()
        union = np.logical_or(preds, targets).sum()
        total = targets.size

        dice = (2 * inter + 1e-5) / (preds.sum() + targets.sum() + 1e-5)
        acc = (preds == targets).sum() / total

        return {"dice": dice, "accuracy": acc}

# -------------------- 分类任务指标 --------------------
class ClassificationMetrics:
    def __init__(self):
        pass

    def compute(self, preds, labels):
        preds = preds.asnumpy().argmax(axis=1)
        labels = labels.asnumpy()

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average='macro', zero_division=0)
        rec = recall_score(labels, preds, average='macro', zero_division=0)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
