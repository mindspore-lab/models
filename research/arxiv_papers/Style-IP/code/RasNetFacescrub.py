import os
import mindspore
from mindspore import nn, Tensor
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
import mindspore.dataset as ds
from mindspore.dataset.vision import ImageFolderDataset, Resize, ToTensor, Normalize
import mindspore.context as context
from mindspore.common import dtype as mstype

# 设置MindSpore环境
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# 定义ResNet-18模型
class BasicBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Cell):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Dense(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        x = self.fc(x)
        return x

# 数据加载和预处理
def load_facescrub_data(batch_size=32, dataset_path="./FaceScrub"):
    # 使用 ImageFolderDataset 加载图像数据
    # 假设数据集结构为：
    # FaceScrub/
    # ├── train/
    # │   ├── class1/
    # │   ├── class2/
    # │   └── ...
    # └── test/
    #     ├── class1/
    #     ├── class2/
    #     └── ...

    # 加载训练数据集
    train_ds = ImageFolderDataset(os.path.join(dataset_path, "train"), num_parallel_workers=8, shuffle=True)

    # 加载测试数据集
    test_ds = ImageFolderDataset(os.path.join(dataset_path, "test"), num_parallel_workers=8, shuffle=False)

    # 图像预处理：调整大小，转换为张量，归一化
    transform = [
        Resize((224, 224)),  # ResNet-18通常输入尺寸为224x224
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 常见的图像归一化方法
    ]
    train_ds = train_ds.map(operations=transform, input_columns=["image"])
    test_ds = test_ds.map(operations=transform, input_columns=["image"])

    train_ds = train_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    return train_ds, test_ds

# 训练模型
def train_model():
    # 创建ResNet18模型实例
    resnet18 = ResNet18(num_classes=1000)  # 假设FaceScrub数据集有1000个类
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')  # 交叉熵损失
    optimizer = nn.Adam(params=resnet18.trainable_params(), learning_rate=0.001)

    # 加载数据集
    train_dataset, test_dataset = load_facescrub_data(batch_size=32, dataset_path="./FaceScrub")

    # 创建MindSpore模型
    model = Model(resnet18, loss_fn=loss_fn, optimizer=optimizer, metrics={"accuracy"})

    # 训练回调函数：监控损失
    loss_monitor = LossMonitor(per_print_times=100)

    # 开始训练
    model.train(epoch=10, train_dataset=train_dataset, callbacks=[loss_monitor], dataset_sink_mode=False)

    # 评估模型
    model.eval(test_dataset, dataset_sink_mode=False)

# 运行训练
if __name__ == "__main__":
    train_model()
