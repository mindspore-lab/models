import numpy as np
import mindspore as ms
from utils import setup_logging
from trainer import PM25Trainer
import logging


def main():
    setup_logging()

    try:
        # 设置设备上下文
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_device('CPU')  # 替代旧的device_target参数
        # 初始化训练器
        trainer = PM25Trainer(
            r"D:\pythonpro\MindSpore\project 3\dataset\FiveCitiePMData\BeijingPM20100101_20151231.csv"
        )

        # 训练模型
        model, train_history, val_history = trainer.train()

        # 保存模型和训练记录
        ms.save_checkpoint(model, "pm25_model_beijing.ckpt")
        np.save("train_loss.npy", train_history)
        np.save("val_loss.npy", val_history)

        logging.info(f"模型训练完成，最终训练损失: {train_history[-1]:.4f}, 验证损失: {val_history[-1]:.4f}")
    except Exception as e:
        logging.error(f"程序运行出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()