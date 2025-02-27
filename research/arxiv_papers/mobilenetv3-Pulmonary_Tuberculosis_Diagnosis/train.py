import mindspore as ms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.transforms.c_transforms as c_transforms
from mindspore.dataset import ImageFolderDataset
from model import MobileNetV3
import mindspore.nn as nn
from mindspore.train import Model, LossMonitor, Accuracy
import os
import shutil
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, auc, precision_recall_curve, classification_report
import numpy as np
from tqdm import tqdm 
import seaborn as sns

ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE, jit_config={"jit_level":"O2"}, ascend_config={"precision_mode":"allow_mix_precision"})

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    normal_dir = os.path.join(data_dir, 'Normal')
    tuberculosis_dir = os.path.join(data_dir, 'Tuberculosis')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # 创建目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'Normal'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'Tuberculosis'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'Normal'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'Tuberculosis'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'Normal'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'Tuberculosis'), exist_ok=True)

    # 获取文件列表
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)]
    tuberculosis_files = [os.path.join(tuberculosis_dir, f) for f in os.listdir(tuberculosis_dir)]

    # 打乱文件列表
    random.shuffle(normal_files)
    random.shuffle(tuberculosis_files)

    def split_files(files, train_size, val_size):
        train_files = files[:train_size]
        val_files = files[train_size:train_size + val_size]
        test_files = files[train_size + val_size:]
        return train_files, val_files, test_files

    # 计算划分数量
    num_normal = len(normal_files)
    num_tuberculosis = len(tuberculosis_files)
    train_normal_size = int(num_normal * train_ratio)
    val_normal_size = int(num_normal * val_ratio)
    test_normal_size = num_normal - train_normal_size - val_normal_size
    train_tuberculosis_size = int(num_tuberculosis * train_ratio)
    val_tuberculosis_size = int(num_tuberculosis * val_ratio)
    test_tuberculosis_size = num_tuberculosis - train_tuberculosis_size - val_tuberculosis_size

    # 划分正常图像
    train_normal, val_normal, test_normal = split_files(normal_files, train_normal_size, val_normal_size)
    # 划分肺结核图像
    train_tuberculosis, val_tuberculosis, test_tuberculosis = split_files(tuberculosis_files, train_tuberculosis_size, val_tuberculosis_size)

    def copy_files(files, dst_dir):
        for file in files:
            shutil.copy(file, dst_dir)

    # 复制正常图像
    copy_files(train_normal, os.path.join(train_dir, 'Normal'))
    copy_files(val_normal, os.path.join(val_dir, 'Normal'))
    copy_files(test_normal, os.path.join(test_dir, 'Normal'))
    # 复制肺结核图像
    copy_files(train_tuberculosis, os.path.join(train_dir, 'Tuberculosis'))
    copy_files(val_tuberculosis, os.path.join(val_dir, 'Tuberculosis'))
    copy_files(test_tuberculosis, os.path.join(test_dir, 'Tuberculosis'))


def preprocess_dataset(dataset_dir, batch_size=32, image_size=(512, 512), train=True):
    # 加载数据集
    dataset = ImageFolderDataset(dataset_dir, num_parallel_workers=4)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    if train:
        trans = [
            c_vision.RandomCropDecodeResize(image_size),
            c_vision.RandomHorizontalFlip(prob=0.5),
            c_vision.Normalize(mean=mean, std=std),
            c_vision.HWC2CHW()
        ]
    else:
        trans = [
            c_vision.Decode(),
            c_vision.Resize(image_size),
            c_vision.CenterCrop(image_size),
            c_vision.Normalize(mean=mean, std=std),
            c_vision.HWC2CHW()
        ]
    type_cast_op = c_transforms.TypeCast(ms.int32)
    dataset = dataset.map(operations=trans, input_columns=["image"], num_parallel_workers=4)
    dataset = dataset.map(operations=type_cast_op, input_columns=["label"], num_parallel_workers=4)
    dataset = dataset.batch(batch_size)
    return dataset


def load_datasets(batch_size=32):
    train_dataset = preprocess_dataset("dataset/train", batch_size=batch_size, image_size=(512, 512), train=True)
    val_dataset = preprocess_dataset("dataset/val", batch_size=batch_size, image_size=(512, 512), train=False)
    test_dataset = preprocess_dataset("dataset/test", batch_size=batch_size, image_size=(512, 512), train=False)
    return train_dataset, val_dataset, test_dataset


def create_model():
    model = MobileNetV3(num_classes=2)  # 二分类任务：正常 vs 肺结核
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    opt = nn.Momentum(model.trainable_params(), learning_rate=0.01, momentum=0.9)
    return model, loss, opt


def train_and_evaluate(epochs=10, lr=0.001, batch_size=32, save_dir="graphs"):
    os.makedirs(save_dir, exist_ok=True)
    # 划分数据集
    split_dataset('dataset')
    train_dataset, val_dataset, test_dataset = load_datasets(batch_size)
    # 创建模型和优化器
    model = MobileNetV3(num_classes=2)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(model.trainable_params(), learning_rate=lr, momentum=0.9)
    # 封装训练网络
    net_with_loss = nn.WithLossCell(model, loss_fn)
    train_net = nn.TrainOneStepCell(net_with_loss, opt)
    # 记录训练指标
    train_loss_history = []
    val_accuracy_history = []
    best_val_accuracy = 0.0
    best_model_path = None
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.set_train(True)  # 关键修复：使用 set_train(True)
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataset.create_dict_iterator(), 
                            desc=f"Epoch {epoch+1}/{epochs}", 
                            total=len(train_dataset),
                            bar_format="{l_bar}{bar:20}{r_bar}")
        for batch in progress_bar:
            data = batch["image"]
            label = batch["label"]
            loss = train_net(data, label)
            epoch_loss += loss.asnumpy().item()
            progress_bar.set_postfix({"Batch Loss": loss.asnumpy().item()})
        avg_epoch_loss = epoch_loss / len(train_dataset)
        train_loss_history.append(avg_epoch_loss)

        # 验证阶段
        model.set_train(False)  # 关键修复：使用 set_train(False)
        acc_metric = Accuracy()
        val_progress = tqdm(val_dataset.create_dict_iterator(), 
                            desc="Validating", 
                            total=len(val_dataset), 
                            leave=False)
        for batch in val_progress:
            data = batch["image"]
            label = batch["label"]
            output = model(data)
            acc_metric.update(output, label)
        val_accuracy = acc_metric.eval()
        val_accuracy_history.append(val_accuracy)

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            if best_model_path: 
                os.remove(best_model_path)
            best_val_accuracy = val_accuracy
            best_model_path = f"best_model_epoch_{epoch+1}.ckpt"
            ms.save_checkpoint(model, best_model_path)

        # 打印日志
        print(f"\nEpoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_epoch_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}")

    # 训练结束后绘制完整的 loss 和 accuracy 图
    plt.figure(figsize=(12, 5))
    
    # 绘制训练 loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_loss_history, label='Train Loss', color='blue')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 绘制验证 accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accuracy_history, label='Validation Accuracy', color='orange')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # 保存最终的训练结果图
    plt.savefig(os.path.join(save_dir, 'final_training_progress.png'), bbox_inches='tight')
    plt.close()

    # 最终评估
    generate_evaluation_plots(best_model_path, test_dataset, save_dir)


def generate_evaluation_plots(model_path, test_dataset, save_dir):
    # 加载模型
    model = MobileNetV3(num_classes=2)
    ms.load_checkpoint(model_path, model, strict_load=True)
    model.set_train(False)  # 关键修复：使用 set_train(False)
    
    # 检查测试数据
    if test_dataset.get_dataset_size() == 0:
        raise ValueError("测试数据集为空！")
    
    # 收集预测结果
    true_labels, pred_labels, probs = [], [], []
    for batch in test_dataset.create_dict_iterator():
        images = batch["image"]
        labels = batch["label"]
        outputs = model(images)
        preds = outputs.argmax(axis=1)
        prob = ms.ops.Softmax(axis=1)(outputs)[:, 1].asnumpy()
        true_labels.extend(labels.asnumpy())
        pred_labels.extend(preds.asnumpy())
        probs.extend(prob)
    
    # 1. 混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Tuberculosis'], 
                yticklabels=['Normal', 'Tuberculosis'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()
    
    # 2. ROC曲线
    fpr, tpr, _ = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), bbox_inches='tight')
    plt.close()
    
    # 3. PR曲线
    precision, recall, _ = precision_recall_curve(true_labels, probs)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='PR Curve')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), bbox_inches='tight')
    plt.close()
    
    # 4. 分类报告热力图
    report = classification_report(true_labels, pred_labels, target_names=['Normal', 'Tuberculosis'], output_dict=True)
    plt.figure(figsize=(8, 4))
    sns.heatmap([[report['Normal']['precision'], report['Normal']['recall']],
                 [report['Tuberculosis']['precision'], report['Tuberculosis']['recall']]],
                annot=True, cmap='Blues', fmt='.2f',
                xticklabels=['Precision', 'Recall'], 
                yticklabels=['Normal', 'Tuberculosis'])
    plt.title('Classification Report Heatmap')
    plt.savefig(os.path.join(save_dir, 'classification_heatmap.png'), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    train_and_evaluate(
        epochs=10, 
        lr=0.001,  # 适当增大学习率
        batch_size=16,
        save_dir="graphs"
    )
    