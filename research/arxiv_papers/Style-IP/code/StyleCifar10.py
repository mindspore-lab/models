import mindspore
import numpy as np
from mindspore import nn, Tensor, ops
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
import mindspore.dataset as ds
from mindspore.dataset.vision import Decode, RandomCrop, Normalize, HWC2CHW
import mindspore.context as context
import mindspore.nn.probability as mnp
from mindspore import Parameter
import mindspore.ops as ops

# 设置MindSpore环境
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# 定义一个简化的VGG-like网络来提取风格和内容特征
class VGG19(nn.Cell):
    def __init__(self):
        super(VGG19, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

    def construct(self, x):
        x = ops.relu(self.conv1_1(x))
        x = ops.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = ops.relu(self.conv2_1(x))
        x = ops.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = ops.relu(self.conv3_1(x))
        x = ops.relu(self.conv3_2(x))
        x = ops.relu(self.conv3_3(x))
        x = self.pool3(x)
        x = ops.relu(self.conv4_1(x))
        x = ops.relu(self.conv4_2(x))
        x = ops.relu(self.conv4_3(x))
        x = self.pool4(x)
        return x


# 定义损失函数：内容损失 + 风格损失
class LossFunctions(nn.Cell):
    def __init__(self, vgg19):
        super(LossFunctions, self).__init__()
        self.vgg19 = vgg19
        self.mse_loss = nn.MSELoss()

    def construct(self, content_image, style_image, generated_image):
        # 计算内容损失
        content_features = self.vgg19(content_image)
        generated_features = self.vgg19(generated_image)
        content_loss = self.mse_loss(content_features, generated_features)

        # 计算风格损失：通过 Gram Matrix 计算
        style_features = self.vgg19(style_image)
        style_generated_features = self.vgg19(generated_image)
        
        style_loss = self.mse_loss(style_features, style_generated_features)

        # 总损失 = 内容损失 + 风格损失
        total_loss = content_loss + style_loss
        return total_loss


# CIFAR-10 数据集加载和预处理
def load_cifar10_data(batch_size=32):
    train_ds = ds.Cifar10Dataset("/path/to/cifar10/train")
    transform = [
        Decode(),
        RandomCrop(32),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        HWC2CHW()
    ]
    train_ds = train_ds.map(operations=transform, input_columns=["image"])
    train_ds = train_ds.batch(batch_size)
    return train_ds


# 设置优化器
optimizer = nn.Adam(params=None, learning_rate=0.001)

# 初始化VGG网络和损失函数
vgg19 = VGG19()
loss_fn = LossFunctions(vgg19)

# 训练模型
def train_style_transfer_model(train_dataset):
    for epoch in range(10):  # 训练 10 个 epoch
        for batch in train_dataset.create_tuple_iterator():
            content_image, style_image = batch
            content_image = Tensor(content_image)
            style_image = Tensor(style_image)

            # 生成图像：用随机噪声初始化生成图像
            generated_image = ops.Zeros()(content_image.shape)

            # 计算损失
            loss = loss_fn(content_image, style_image, generated_image)

            # 反向传播并更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 输出损失值
            print(f"Epoch [{epoch + 1}], Loss: {loss.asnumpy()}")

# 加载数据并训练模型
train_dataset = load_cifar10_data()
train_style_transfer_model(train_dataset)
