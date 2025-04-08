import numpy as np
import mindspore as ms
from mindspore import nn, dataset
from models import PM25PredictionModel
from pm25_preprocessor import PM25Preprocessor
from utils import clear_memory
import logging


class PM25Trainer:
    """PM2.5模型训练器"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.train_loss_history = []
        self.val_loss_history = []

    def _create_datasets(self, train_data, val_data):
        """创建MindSpore数据集"""
        train_ds = dataset.NumpySlicesDataset(
            (train_data[:, :, :-1], train_data[:, -1, -1]),
            column_names=["data", "label"],
            shuffle=True
        ).batch(64, drop_remainder=True)

        val_ds = dataset.NumpySlicesDataset(
            (val_data[:, :, :-1], val_data[:, -1, -1]),
            column_names=["data", "label"],
            shuffle=False
        ).batch(64, drop_remainder=True)

        return train_ds, val_ds

    def _cosine_decay(self, current_step, initial_lr, min_lr, decay_steps):
        """余弦退火学习率调度"""
        decay = 0.5 * (1 + np.cos(np.pi * current_step / decay_steps))
        return min_lr + (initial_lr - min_lr) * decay

    def _forward_fn(self, data, label):
        """前向计算函数"""
        pred = self.model(data)
        return nn.MSELoss()(pred, label)

    def _evaluate(self, val_ds, loss_fn):
        """评估模型"""
        val_loss = 0
        for batch in val_ds.create_tuple_iterator():
            val_loss += loss_fn(self.model(batch[0]), batch[1]).asnumpy()
        return val_loss / len(val_ds)

    def train(self, epochs=30, patience=10):
        """训练主循环"""
        clear_memory()
        ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

        # 数据预处理
        logging.info("开始数据预处理")
        preprocessor = PM25Preprocessor()
        train_data, val_data = preprocessor.transform(self.data_path)
        train_ds, val_ds = self._create_datasets(train_data, val_data)

        # 模型初始化
        self.model = PM25PredictionModel(feat_dim=len(preprocessor.feature_columns))

        # 训练配置
        initial_lr = 1e-5
        min_lr = 1e-7
        decay_steps = 10 * len(train_ds)
        optimizer = nn.Adam(self.model.trainable_params(), learning_rate=initial_lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        grad_fn = ms.value_and_grad(self._forward_fn, None, optimizer.parameters)

        # 早停机制
        best_val_loss = float('inf')
        counter = 0

        for epoch in range(epochs):
            # 更新学习率
            current_lr = self._cosine_decay(epoch, initial_lr, min_lr, decay_steps)
            optimizer.learning_rate.set_data(ms.Tensor(current_lr, ms.float32))

            # 训练阶段
            self.model.set_train()
            epoch_loss = 0
            for batch in train_ds.create_tuple_iterator():
                loss = grad_fn(batch[0], batch[1])
                optimizer(loss[1])
                epoch_loss += loss[0].asnumpy()

            train_loss = epoch_loss / len(train_ds)
            self.train_loss_history.append(train_loss)

            # 验证阶段
            val_loss = self._evaluate(val_ds, loss_fn)
            self.val_loss_history.append(val_loss)

            # 早停判断和模型保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                ms.save_checkpoint(self.model, "best_model.ckpt")
                logging.info(f"Epoch {epoch + 1}: 发现更好的模型，验证损失: {best_val_loss:.4f}")
            else:
                counter += 1
                if counter >= patience:
                    logging.info(f"早停触发，连续{patience}个epoch验证损失未改善")
                    break

            logging.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Patience: {counter}/{patience}"
            )

        # 加载最佳模型
        param_dict = ms.load_checkpoint("best_model.ckpt")
        ms.load_param_into_net(self.model, param_dict)

        return self.model, self.train_loss_history, self.val_loss_history