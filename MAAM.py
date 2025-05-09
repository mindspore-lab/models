import os
import pickle
import numpy as np
import time  # 导入time模块
import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.dataset import GeneratorDataset
from mindspore.dataset import vision, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 设置CPU环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 数据加载函数（优化版）
def load_cifar10(data_path, batch_files):
    images = []
    labels = []
    
    for batch_file in batch_files:
        with open(os.path.join(data_path, batch_file), 'rb') as fo:
            batch_data = pickle.load(fo, encoding='bytes')
        
        # 调整数据形状为NHWC格式
        batch_images = batch_data[b'data'].reshape(-1, 3, 32, 32)
        batch_images = batch_images.transpose(0, 2, 3, 1)  # 转换为NHWC格式
        images.append(batch_images)
        labels.extend(batch_data[b'labels'])
    
    return np.concatenate(images), np.array(labels)

class CIFAR10Dataset:
    def __init__(self, data_path, is_train=True):
        if is_train:
            self.batch_files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            self.batch_files = ["test_batch"]
            
        self.images, self.labels = load_cifar10(data_path, self.batch_files)
        
        # 数据增强管道
        self.transform = transforms.Compose([ 
            vision.RandomCrop(32, padding=4),
            vision.RandomHorizontalFlip(),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.HWC2CHW()  # 转换为CHW格式
        ])
    
    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        
        # 应用数据增强
        img = self.transform(img)
        return img.astype(np.float32), label.astype(np.int32)
    
    def __len__(self):
        return len(self.labels)

import mindspore as ms
from mindspore import nn, ops

class AgentBlock(nn.Cell):
    """
    多层代理模块：提取不同尺度的局部特征。
    """
    def __init__(self, in_channels, out_channels):
        super(AgentBlock, self).__init__()
        self.block = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

    def construct(self, x):
        return self.block(x)

class AgentAttentionFusion(nn.Cell):
    """
    轻量版 Agent Attention：为每个 agent 输出分配一个可学习权重，然后加权融合。
    """
    def __init__(self, num_agents=3):
        super(AgentAttentionFusion, self).__init__()
        self.attn_weights = ms.Parameter(ops.ones((num_agents, 1, 1, 1), ms.float32))
        self.softmax = nn.Softmax(axis=0)

    def construct(self, agents):
        weights = self.softmax(self.attn_weights)  # [3,1,1,1] 权重归一化
        fused = sum(w * f for w, f in zip(weights, agents))  # 加权融合
        return fused

class MAAModule(nn.Cell):
    """
    多代理聚合模块（MAAM）：组合多个代理特征图并使用 Agent Attention 机制融合。
    """
    def __init__(self):
        super(MAAModule, self).__init__()
        self.agent1 = AgentBlock(3, 64)
        self.agent2 = AgentBlock(3, 64)
        self.agent3 = AgentBlock(3, 64)
        self.fusion = AgentAttentionFusion(num_agents=3)
        self.reduce = nn.SequentialCell([
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])

    def construct(self, x):
        a1 = self.agent1(x)
        a2 = self.agent2(x)
        a3 = self.agent3(x)
        fused = self.fusion([a1, a2, a3])
        out = self.reduce(fused)
        return out

class MAAMNetwork(nn.Cell):
    """
    完整网络结构，包含 MAAM 模块 + 分类头
    """
    def __init__(self, num_classes=10):
        super(MAAMNetwork, self).__init__()
        self.maam = MAAModule()
        self.classifier = nn.SequentialCell([
            nn.Flatten(),
            nn.Dense(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dense(256, num_classes)
        ])

    def construct(self, x):
        x = self.maam(x)
        x = self.classifier(x)
        return x


# 修复后的训练函数
def train():
    # 数据集参数
    data_path = "zhu/datasets/cifar10"  # 修改为实际路径
    batch_size = 64
    
    # 创建数据集
    train_dataset = GeneratorDataset(
        source=CIFAR10Dataset(data_path, is_train=True),
        column_names=["image", "label"],
        shuffle=True,
        num_parallel_workers=4
    ).batch(batch_size)
    
    test_dataset = GeneratorDataset(
        source=CIFAR10Dataset(data_path, is_train=False),
        column_names=["image", "label"],
        shuffle=False,
        num_parallel_workers=4
    ).batch(batch_size)
    
    # 初始化模型
    model = MAAMNetwork()
    
    # 优化器和损失函数
    optimizer = nn.Adam(params=model.trainable_params(), 
                      learning_rate=0.001,
                      weight_decay=0.0001)
    
    loss_fn = nn.CrossEntropyLoss()
    
    # 定义训练步骤
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss
    
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
    
    # 打开文件保存结果
    result_file = open("training_results(main+cnn).txt", "w")
    
    # 训练循环
    for epoch in range(50):
        start_time = time.time()  # 记录epoch开始时间
        model.set_train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch, (data, label) in enumerate(train_dataset):
            loss, grads = grad_fn(data, label)
            optimizer(grads)
            total_loss += loss.asnumpy()
            
            all_preds.extend(ops.argmax(model(data), 1).asnumpy())
            all_labels.extend(label.asnumpy())
            
            if batch % 1 == 0:
                print(f"Epoch [{epoch+1}/50], Step [{batch}/{len(train_dataset)}], Loss: {loss.asnumpy():.4f}")
        
        # 计算训练集的评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        epoch_time = time.time() - start_time  # 计算一个epoch的时间

        # 输出训练结果
        result_file.write(f"Epoch [{epoch+1}/50] Training Metrics:\n")
        result_file.write(f"Loss: {total_loss/len(train_dataset):.4f}\n")
        result_file.write(f"Accuracy: {accuracy:.4f}\n")
        result_file.write(f"Precision: {precision:.4f}\n")
        result_file.write(f"Recall: {recall:.4f}\n")
        result_file.write(f"F1-Score: {f1:.4f}\n")
        result_file.write(f"Training Time: {epoch_time:.4f} seconds\n")  # 输出训练时间
        
        # 验证过程
        model.set_train(False)
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for data, label in test_dataset:
            outputs = model(data)
            predicted = ops.argmax(outputs, 1)
            total += label.shape[0]
            correct += (predicted == label).sum().asnumpy()
            
            all_preds.extend(predicted.asnumpy())
            all_labels.extend(label.asnumpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        # 输出验证结果
        result_file.write(f"Epoch [{epoch+1}/50] Validation Metrics:\n")
        result_file.write(f"Test Accuracy: {accuracy:.4f}\n")
        result_file.write(f"Test Precision: {precision:.4f}\n")
        result_file.write(f"Test Recall: {recall:.4f}\n")
        result_file.write(f"Test F1-Score: {f1:.4f}\n")
        
    result_file.close()

if __name__ == "__main__":
    train()
