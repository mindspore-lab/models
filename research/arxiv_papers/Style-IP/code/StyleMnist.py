import numpy as np
import mindspore
from mindspore import nn, Tensor
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
from mindspore.dataset import GeneratorDataset
import mindspore.ops as ops
from mindspore.dataset.vision import Decode, Normalize, HWC2CHW
import mindspore.context as context
from mindspore import Tensor
import cv2
import os

# 配置设备
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# 加载MNIST数据集
from mindspore.dataset import MnistDataset

def preprocess_data():
    # 使用 MNIST 数据集作为内容图像数据集
    train_ds = MnistDataset('./mnist_train')
    # 对图像进行一些预处理操作，比如归一化
    transform = [HWC2CHW(), Normalize(mean=[0.5], std=[0.5])]
    train_ds = train_ds.map(operations=transform, input_columns=["image"])
    return train_ds

# 定义VGG19网络用于风格迁移
class VGG19(nn.Cell):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg = nn.SequentialCell(
            nn.Conv2d(1, 64, 3, pad_mode="same", weight_init="XavierUniform"),  # C1
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, pad_mode="same", weight_init="XavierUniform"),  # C2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid"),  # P1
            nn.Conv2d(64, 128, 3, pad_mode="same", weight_init="XavierUniform"),  # C3
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, pad_mode="same", weight_init="XavierUniform"),  # C4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid"),  # P2
            nn.Conv2d(128, 256, 3, pad_mode="same", weight_init="XavierUniform"),  # C5
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, pad_mode="same", weight_init="XavierUniform"),  # C6
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid"),  # P3
            nn.Conv2d(256, 512, 3, pad_mode="same", weight_init="XavierUniform"),  # C7
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, pad_mode="same", weight_init="XavierUniform"),  # C8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid")  # P4
        )

    def construct(self, x):
        return self.vgg(x)

# 定义损失函数
class LossFunctions(nn.Cell):
    def __init__(self, vgg19):
        super(LossFunctions, self).__init__()
        self.vgg19 = vgg19
        self.mse_loss = nn.MSELoss()

    def construct(self, content, style, generated):
        # 计算内容损失：使用生成图像和内容图像的VGG19特征进行比较
        content_loss = self.mse_loss(self.vgg19(content), self.vgg19(generated))

        # 计算风格损失：计算风格图像和生成图像的风格损失
        style_loss = self.mse_loss(self.vgg19(style), self.vgg19(generated))

        # 总损失：内容损失 + 风格损失
        total_loss = content_loss + style_loss

        return total_loss

# 初始化网络和损失函数
vgg19 = VGG19()
loss_fn = LossFunctions(vgg19)

# 定义优化器
optimizer = nn.Adam(vgg19.trainable_params(), learning_rate=0.001)

# 加载数据集
train_ds = preprocess_data()

# 定义训练过程
def train(model, train_dataset):
    for epoch in range(20):  # 训练20个epoch
        for batch in train_dataset.create_tuple_iterator():
            content_image, style_image = batch
            content_image = Tensor(content_image)
            style_image = Tensor(style_image)

            # 生成图像：初始化一个随机噪声图像
            generated_image = ops.Zeros()(content_image.shape)

            # 计算损失
            loss = loss_fn(content_image, style_image, generated_image)

            # 反向传播并更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 输出损失
            print(f"Epoch [{epoch + 1}], Loss: {loss.asnumpy()}")


# 创建模型并训练
train(model=loss_fn, train_dataset=train_ds)
