# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [VSE++ 描述](#ADDA描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [在Flickr30K 上训练](#####在MINIST和USPS上训练)
        - [评估性能](#评估性能)
            - [在Flickr30K 上评估](#在USPS上评估)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# VSE++ 描述

VSE++（ Improving Visual-Semantic Embeddings with Hard Negatives）方法于2018年提出，旨在解决跨模态文本检索问题。本文作者在损失函数中加入难例挖掘，使用难例挖掘最小化损失函数，无需任何额外的挖掘成本，实现了性能的提升。

[论文](ttps://link.zhihu.com/?target=http://www.bmva.org/bmvc/2018/contents/papers/0344.pdf)：Faghri F, Fleet D J, Kiros J R, et al. Vse++: Improving visual-semantic embeddings with hard negatives[J]. arXiv preprint arXiv:1707.05612, 2017.

# 模型架构

该模型的主要目标是实现图像与文本之间的匹配，即将图像和相应的文本描述进行关联。它采用了VGG19作为图像编码器和单层RGU作为文本编码器。VGG19是一个经典的卷积神经网络，由19个卷积层和3个全连接层组成。它在大规模图像分类任务上表现出色，并且已被广泛应用于各种计算机视觉任务中。在这个模型中，VGG19负责将输入图像转化为一系列的图像特征向量。单层RGU是一种基于循环神经网络（Recurrent Neural Network）的编码器。它可以对输入的文本序列数据进行建模，并输出一个固定长度的向量表示。


# 数据集
该模型使用了Flickr30K 数据集。Flickr30K 是一个常用的图像描述数据集，包含了30,000个来自Flickr网站的图像，每个图像都有对应的文本描述。这个数据集被广泛用于图像与文本之间的关联学习任务，如图像描述生成和图像检索等。请使用以下命令下载数据集并解压放在`.data/`文件夹

```
wget http://www.cs.toronto.edu/~faghri/vsepp/data.tar
```
解压成功后的`.data/`文件夹应包含如下文件夹：f30k，f30k_precomp

- 数据集大小：Flickr30K数据集包含了大约31,000张图像，每张图像对应5个描述标注，总共约15万个标注。
- 数据格式：npy和txt文件
# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门
通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：
- Ascend处理器环境运行

  ```python
  # 运行训练示例
  python train.py --device_id 0  > train_Ascend.log 2>&1 &
  或
  bash scripts/run_train.sh
   # 运行评估示例
  python eval.py --device_id 0  > eval_Ascend.log 2>&1
  或
  bash scripts/run_eval.sh
  ```

- CPU处理器环境运行



  ```python
  # 运行训练示例
  bash run_train_cpu.sh
  或
  python train.py --device "CPU"  > train_CPU.log 2>&1 &

  # 运行评估示例
  bash run_eval_cpu.sh
  或
  python eval.py --device "CPU"  > eval_CPU.log 2>&1 &
  ```
# 脚本说明
## 脚本及样例代码

  ```text
├── vsepp  
    |—— data
        |——f30k_precomp             # f30k_precomp数据集
    |—— src
        |——data.py                  # 处理f30k_precomp数据集
        |——trainer.py                # 构建训练器
|——vocab.py                 # 处理文本编码
|——f30k_precomp_vocab.pkl    # 数据集图像文本对应关系
    |── models
        |──model.py         		 # 主要模型结构
        |──model_vgg19.py           # vgg19模型结构
        |──var_init.py                # 模型权重初始化方法
|── model_utils
        |——config.py                 # 处理配置参数
        |——device_adapter.py          # 获取云ID
        |——local_adapter.py           # 获取本地ID
        |——moxing_adapter.py         # 参数处理
    ├── scripts
        ├──run_eval.sh              # Ascend评估的shell脚本
        ├──run_eval_cpu.sh          # CPU处理器评估的shell脚本
        ├──run_train.sh              # 用于Ascend训练的shell脚本
        ├──run_train_cpu.sh          # 用于CPU训练的shell脚本
├── default_config.yaml          # 参数配置文件
    ├── eval.py                    # 评估脚本
    ├── export.py                   # 将checkpoint文件导出到air/mindir
    ├── README.md                # vsepp相关英文说明
    ├── README_CN.md             # vsepp相关中文说明
    ├── requirements.txt             # 需要的包
    ├── train.py                    # 训练脚本

  ```

## 脚本参数

在default_config.yaml中可以同时配置训练参数和评估参数。

- 配置VSE++。
```python
batch_size: 128
epoch: 30
learning rate: 0.0002
lr_update: 15
optimizer: Adam
margin: 0.2
word_dim: 300
embed_size: 1024
grad_clip: 2
crop_size: 224
num_layers: 1
max_violation: true
img_dim: 4096
workers: 1
log_step: 200
val_step: 500
cnn_type: "vgg19"
use_abs: false
no_imgnorm: false
seed: 123
device_id: 7
device_target: "Ascend"
data_path: "data"
data_name: "f30k_precomp"
vocab_path: "./vocab/"
num_epochs: 30
use_restval: false
enable_modelarts: false
finetune: false
ckpt_file: "./best.ckpt"
vocab_size: 8481
file_name: "model"
file_format: "MINDIR"
```

更多配置细节请参考脚本`default_config.yaml`。  

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py --device_id 0  > train_Ascend.log 2>&1 &
  或
  bash scripts/run_train.sh
  ```

  训练结束后，您可得到结果如下：
  

```python
================epoch :30================
[2024-03-09 10:35:59] step: [1/1132] loss: 12.309782 lr: 0.00002
[2024-03-09 10:36:19] step: [201/1132] loss: 9.338764 lr: 0.00002
[2024-03-09 10:36:38] step: [401/1132] loss: 9.604823 lr: 0.00002
[2024-03-09 10:36:58] step: [601/1132] loss: 9.094549 lr: 0.00002
[2024-03-09 10:37:17] step: [801/1132] loss: 9.84281 lr: 0.00002
[2024-03-09 10:37:36] step: [1001/1132] loss: 10.121641 lr: 0.00002
Per step costs time(ms): 0:00:00.096494
0/39
10/39
20/39
30/39
Image to text: 26.7, 52.4, 61.6, 5.0, 42.1
Text to image: 20.2, 42.3, 52.6, 9.0, 63.6
rsum:  255.8517034068136
Best score: 299.47895791583164
```
## 评估过程

### 评估  

- 在Ascend环境运行评估
  ```bash
  python eval.py --device_id 0  > eval_Ascend.log 2>&1
  或
  bash scripts/run_eval.sh
  ```

 上述所有命令将在后台运行，您可以通过log文件查看结果。测试数据集的准确性如下：
```python
rsum: 295.3
Average i2t Recall: 53.6
Image to text: 31.7 57.9 71.1 4.0 26.9
Average t2i Recall: 44.9
Text to image: 23.4 49.4 61.8 6.0 34.8
```
## 导出过程

### 导出

  ```shell
  python export.py
  ```

# 模型描述

## 性能

### 训练性能

#### Flickr30K 在VSE++的性能

|         参数         |                                            Ascend                                            |
|:--------------------------:|:--------------------------------------------------------------------------------------------:|
|       模型版本        |                                            VSE++                                           |
|          资源          |                                  Ascend 910；CPU 2.60GHz，72核；内存 503G;                                  |
|       上传日期        |                                          2023-03-18                                          |
|     MindSpore版本      |                                            1.10.1                                             |
|          数据集           |                                     Flickr30K                                     |
|    训练参数     | batch_size=128；epoch=30；learning rate=0.0002；loss function= ContrastiveLoss；optimizer=Adam；margin=0.2；word_dim=300；embed_size=1024；grad_clip=2；crop_size=224；num_layers=1；max_violation=true；img_dim=4096|
|         优化器          |                                             Adam                                             |
|       损失函数        |                                         Max Hinge Loss                                          |
|          输出           |                                              召回率                                              |
|           速度            |                                 95毫秒/步 (单卡）                             
|         总时长         |                                         48.5分钟（单卡）                                        |                                       |                                |
|    推理模型     |                               41.78M(.air文件)， 41.76(.mindir文件）                              |

# 随机情况说明

在train.py中，我们使用utiles.py中的init_random_seed()函数设置了随机数种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
