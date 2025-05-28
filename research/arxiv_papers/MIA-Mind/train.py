import os
import argparse
import mindspore as ms
from mindspore import context, nn, ops

from data import create_isbi_dataset, get_cifar10_dataset, load_cic_data, create_cic_dataset
from model.ResNet_50 import ResNetCBAM, ResidualBlock
from model.U_Net import UNetCBAM
from model.cnn import CNN_mia
from metrics import UnetMetrics, ClassificationMetrics

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# -------------------- 损失函数 --------------------
class DiceLoss(nn.Cell):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.smooth = ms.Tensor(1e-5, ms.float32)

    def construct(self, logits, targets):
        preds = self.sigmoid(logits)
        inter = ops.ReduceSum()(preds * targets)
        union = ops.ReduceSum()(preds + targets)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice

class ClassificationLoss(nn.Cell):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, logits, labels):
        return self.loss(logits, labels)

# -------------------- 训练逻辑 --------------------
def train_model(model, train_ds, val_ds, loss_fn, metrics_fn, epochs=10, lr=0.001):
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)
    train_net = nn.TrainOneStepCell(nn.WithLossCell(model, loss_fn), optimizer)
    train_net.set_train()

    for epoch in range(epochs):
        total_loss = 0
        steps = 0

        for batch in train_ds.create_dict_iterator():
            inputs = batch.get("image") or batch.get("feature")
            labels = batch["label"]

            loss = train_net(inputs, labels)
            total_loss += loss.asnumpy()
            steps += 1

        print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {total_loss/steps:.4f}")

        evaluate(model, val_ds, metrics_fn)

# -------------------- 验证逻辑 --------------------
def evaluate(model, dataset, metrics_fn):
    model.set_train(False)
    all_preds = []
    all_labels = []

    for batch in dataset.create_dict_iterator():
        inputs = batch.get("image") or batch.get("feature")
        labels = batch["label"]
        logits = model(inputs)

        all_preds.append(logits)
        all_labels.append(labels)

    preds = ops.Concat()(all_preds)
    labels = ops.Concat()(all_labels)

    results = metrics_fn.compute(preds, labels)
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='cifar10', choices=['cifar10', 'isbi2012', 'cic'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    if args.task == 'cifar10':
        train_ds, val_ds = get_cifar10_dataset(batch_size=args.batch_size)
        model = ResNetCBAM(ResidualBlock, [3, 4, 6, 3], num_classes=10)
        loss_fn = ClassificationLoss()
        metrics_fn = ClassificationMetrics()

    elif args.task == 'isbi2012':
        train_ds = create_isbi_dataset(batch_size=args.batch_size, shuffle=True)
        val_ds = create_isbi_dataset(batch_size=args.batch_size, shuffle=False)
        model = UNetCBAM()
        loss_fn = DiceLoss()
        metrics_fn = UnetMetrics(threshold=0.5)

    elif args.task == 'cic':
        X_train, X_test, y_train, y_test = load_cic_data()
        train_ds = create_cic_dataset(X_train, y_train, batch_size=args.batch_size)
        val_ds = create_cic_dataset(X_test, y_test, batch_size=args.batch_size)
        model = CNN_mia(in_channels=X_train.shape[1])
        loss_fn = ClassificationLoss()
        metrics_fn = ClassificationMetrics()

    print(f"\n🚀 Starting training for task: {args.task}")
    train_model(model, train_ds, val_ds, loss_fn, metrics_fn, epochs=args.epochs, lr=args.lr)
