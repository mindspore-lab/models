import gc
import psutil
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from sklearn.model_selection import train_test_split
from data_preprocessing import load_nf_bot_iot_data
from data_loader import create_dataloader
from model import CyberDefenseModel
from train_eval import AdaptiveTrainer, evaluate, generate_predictions_csv

def main():
    # 内存监控
    print(f"初始内存: {psutil.virtual_memory().used // 1024 // 1024} MB")

    # 数据加载
    try:
        features, labels, label_encoder = load_nf_bot_iot_data(r"D:\pythonpro\MindSpore\data\archive\NF-BoT-IoT.parquet")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit(1)

    # 数据集划分
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    except ValueError as e:
        print(f"数据集划分错误: {str(e)}")
        exit(1)

    # 创建数据加载器
    seq_length = 10
    batch_size = 256
    try:
        train_loader = create_dataloader(X_train, y_train, seq_length=seq_length, batch_size=batch_size)
        val_loader = create_dataloader(X_val, y_val, seq_length=seq_length, batch_size=batch_size, shuffle=False)
        test_loader = create_dataloader(X_test, y_test, seq_length=seq_length, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"数据加载器创建失败: {str(e)}")
        exit(1)

    # 初始化模型
    model = CyberDefenseModel(input_dim=X_train.shape[1])
    loss_fn = nn.BCELoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.0001)
    trainer = AdaptiveTrainer(model, loss_fn, optimizer)

    # 训练循环
    best_f1 = 0
    for epoch in range(20):
        model.set_train()
        epoch_loss = 0
        for batch_data, batch_labels in train_loader:
            loss = trainer.train_step(batch_data, batch_labels)
            epoch_loss += loss.asnumpy()

        if (epoch + 1) % 5 == 0:
            trainer.adjust_threshold(val_loader)
            val_metrics = evaluate(model, val_loader)

            if val_metrics['F1'] > best_f1:
                best_f1 = val_metrics['F1']
                ms.save_checkpoint(model, "best_model.ckpt")

            print(f"[Epoch {epoch + 1}] 阈值: {model.threshold.asnumpy()[0]:.4f} | "
                  f"验证集 F1: {val_metrics['F1']:.4f}")

        print(f"Epoch {epoch + 1}: 训练损失={epoch_loss / len(train_loader):.4f}")

    # 最终测试
    ms.load_param_into_net(model, ms.load_checkpoint("best_model.ckpt"))
    test_metrics = evaluate(model, test_loader)
    print("\n=== 测试结果 ===")
    print(f"准确率: {test_metrics['Accuracy']:.4f}")
    print(f"精确率: {test_metrics['Precision']:.4f}")
    print(f"召回率: {test_metrics['Recall']:.4f}")
    print(f"F1分数: {test_metrics['F1']:.4f}")

    # 生成预测结果的CSV文件
    generate_predictions_csv(model, test_loader, label_encoder, output_path="test_predictions.csv")

    # 训练结束后清理内存
    gc.collect()

if __name__ == "__main__":
    main()