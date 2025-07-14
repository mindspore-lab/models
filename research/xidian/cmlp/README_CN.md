# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [ConvMLP描述](#ConvMLP描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
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
        - [评估性能](#评估性能)
            - [CIFAR-10上评估CMLP](#CIFAR-10上评估CMLP)
            - [CIFAR-100上评估CMLP](#CIFAR-100上评估CMLP)


<!-- /TOC -->

# ConvMLP描述
2021.9.18，UO&UIUC 提出了 ConvMLP：一个用于视觉识别的层次卷积MLP。作为一个轻量级(light-weight)、阶段级(stage-wise)、具备卷积层和MLP的联合设计(co-design)，ConvMLP 在 ImageNet-1k 上以仅仅 2.4G MACs 和 9M 参数量 
(分别是 MLP-Mixer-B/16 的 15% 和 19%) 达到了 76.8% 的 Top-1 精度。文章于2023年被CVPR接受。

[论文](https://openaccess.thecvf.com/content/CVPR2023W/WFM/html/Li_ConvMLP_Hierarchical_Convolutional_MLPs_for_Vision_CVPRW_2023_paper.html)：Li J, Hassani A, Walton S, et al. Convmlp: Hierarchical convolutional mlps for vision[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 6306-6315.
# 模型架构

为了解决MLP的框架中对输入维度的约束，作者首先将所有空间mlp替换为跨通道连接 ，并建立一个纯MLP baseline模型。为了弥补空间信息交互，作者在其余的MLP阶段上添加了一个轻量级的卷积阶段 ，并使用卷积层进行降采样 。此外，为了增加MLP阶段的空间连接，作者在每个MLP块中的两个通道MLPs之间添加了一个3×3深度卷积 ，因此称之为Conv-MLP块。作者通过对卷积层和MLP层的共同设计，建立了用于图像分类的ConvMLP模型的原型。
# 数据集

使用的数据集：[CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- 数据集大小：175M，共10个类、6万张32*32彩色图像
    - 训练集：146M，共5万张图像
    - 测试集：29M，共1万张图像
- 数据格式：二进制文件
    - 注：下载的数据将存储到data文件夹下。

使用的数据集：[CIFAR-100](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- 数据集大小：161M，共100个类、6万张32*32彩色图像
    - 训练集：132M，共5万张图像
    - 测试集：29M，共1万张图像
- 数据格式：二进制文件
    - 注：下载的数据将存储到data文件夹下。


# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/CPU）
    - 使用Ascend/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```yaml
  # 添加数据集路径,以训练cifar10为例(相对路径）
  train_data_path:./data/cifar-10-batches-bin
  val_data_path:./data/cifar-10-val-bin

  # 推理前添加checkpoint路径参数(相对路径）
  chcekpoint_path:./checkpoint/cifa10_CMLP.ckpt
  ```

  ```python
  # 运行训练示例
  python train.py

  # 运行评估示例
  python evl.py 

  ```
  为了在CPU端运行，需要将train.py中的device_target参数更改为CPU
- CPU处理器环境运行

  ```python
  ```python
  # 运行训练示例
  python train.py
  # example:parser.add_argument('--device_target', type=str, default='Ascend')->
  # parser.add_argument('--device_target', type=str, default='CPU')

  # 运行评估示例
  python evl.py 

  ```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── CMLP
        ├── README.md                    // CMLP相关说明
        ├── data //数据文件夹
        │   ├──cifar-10-batches-bin
        │   ├──cifar-100-binary
     
        ├── model
        │   ├──mlp.py
        ├── train.py               // 训练脚本
        ├── evl.py               // 评估脚本
        ├── dataset.py       // 数据处理脚本
        ├── pth2ckpt.py            // 将pytorch预训练模型参数传入mindspore模型参数中
```

## 脚本参数

在train.py中可以同时配置训练参数和评估参数。

- 配置CMLP和CIFAR-10数据集。

  ```python
  'run_modelarts':'False'    # 是否使用modelart
  'is_distributed':False   # 是否分布式训练
  'device_id':4            # 选择训练设备
  'batch_size':64          # 训练批次大小
  'epoch_size':125         # 总计训练epoch数
  'dataset_choose':cifar10  # 数据集选择
  'device_target':Ascend     # 硬件选择
  'save_checkpoint_path':./ckpt"     # 模型保存地址
  
  
  ```

- 配置CMLP和CIFAR-100数据集。
  ```python
  'run_modelarts':'False'    # 是否使用modelart
  'is_distributed':False   # 是否分布式训练
  'device_id':4            # 选择训练设备
  'batch_size':64          # 训练批次大小
  'epoch_size':125         # 总计训练epoch数
  'dataset_choose':cifar100  # 数据集选择
  'device_target':Ascend     # 硬件选择
  'save_checkpoint_path':./ckpt"     # 模型保存地址
  
## 导出过程

### 导出

在训练之前需要把把CMLP官方提供的pytorch预训练模型中的参数到处到mindspore模型中。

**注意**：在pth2ckpt.py需要将pytotch模型的相对地址给出，并且CMLP模型的参数大小也需要一一对应。
```shell
python pth2ckpt.py
```

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py 
  ```

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss is " train.log
  epoch:1 step:768, loss is 0.96960
  epcoh:2 step:768, loss is 0.82834
  ...
  ```

  模型检查点保存在当前目录下。

## 评估过程

### 评估

- 在Ascend环境运行时评估CIFAR-10数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/CMLP/checkpoint/cifa10_CMLP.ckpt”。

  ```bash
  python eval.py
  ```

 测试数据集的准确性如下：

  ```bash
  accuracy:{'acc':0.9806}
  ```



# 模型描述

## 性能

### 评估性能

#### CIFAR-10上评估CMLP

| 参数          | Ascend               | GPU             |
| ------------------- |----------------------|-----------------|
| 模型版本       | convmlp_s            | convmlp_s       |
| 资源            | Ascend 910；系统 ubuntu | GPU             |
| 上传日期       | 2023-07-05           | 2023-07-05      |
| MindSpore 版本   | 1.8.1                | 1.8.1           |
| 数据集             | CIFAR-10, 1万张图像      | CIFAR-10, 1万张图像 |
| batch_size          | 128                  | 128             |
| 输出             | 概率                   | 概率              |
| 准确性            | 单卡: 98.06%;          | 单卡：97.64%       |

#### CIFAR-100上评估CMLP

| 参数          | Ascend               | GPU              |
| ------------------- |----------------------|------------------|
| 模型版本       | convmlp_s            | convmlp_s        |
| 资源            | Ascend 910；系统 ubuntu | GPU              |
| 上传日期       | 2023-07-05           | 2023-07-05       |
| MindSpore 版本   | 1.8.1                | 1.8.1            |
| 数据集             | CIFAR-100, 5万张图像     | CIFAR-100, 5万张图像 |
| batch_size          | 128                  | 128              |
| 输出             | 概率                   | 概率               |
| 准确性            | 单卡: 85.12%;          | 单卡：85.48%        |
