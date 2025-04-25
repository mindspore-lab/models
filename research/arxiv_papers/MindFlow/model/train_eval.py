import numpy as np
import pandas as pd
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class AdaptiveTrainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_fn = None

    def forward_fn(self, data, labels):
        preds, _ = self.model(data)
        return self.loss_fn(preds, labels)

    def adjust_threshold(self, val_loader):
        metrics = evaluate(self.model, val_loader, use_threshold=False)
        new_threshold = self.model.threshold * (1 + 0.1 * (metrics['F1'] - 0.9))
        self.model.threshold = ops.clip_by_value(new_threshold, 0.4, 0.7)

    def train_step(self, data, labels):
        if self.grad_fn is None:
            self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters)
        loss, grads = self.grad_fn(data, labels)
        self.optimizer(grads)
        return loss

def evaluate(model, loader, use_threshold=True):
    model.set_train(False)
    y_true, y_pred = [], []
    threshold = model.threshold.asnumpy()[0] if use_threshold else 0.5

    for batch_data, batch_labels in loader:
        preds, _ = model(batch_data)
        y_true.extend(batch_labels.asnumpy().tolist())
        y_pred.extend((preds.asnumpy() > threshold).astype(int).tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return {
        "Accuracy": (TP + TN) / len(y_true),
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    }

def generate_predictions_csv(model, loader, label_encoder, output_path="predictions.csv"):
    model.set_train(False)
    y_true, y_pred = [], []

    for batch_data, batch_labels in loader:
        preds, _ = model(batch_data)
        y_true.extend(batch_labels.asnumpy().tolist())
        y_pred.extend((preds.asnumpy() > 0.5).astype(int).tolist())

    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    y_true_labels = label_encoder.inverse_transform(y_true)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    df = pd.DataFrame({
        'True_Label': y_true_labels,
        'Predicted_Label': y_pred_labels
    })
    df.to_csv(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")