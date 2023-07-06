# 目录

<!-- TOC -->

* [目录](#目录)
* [MCD描述](#mcd描述)
    * [概述](#概述)
    * [论文](#论文)
* [数据集](#数据集)
* [环境要求](#环境要求)
* [快速入门](#快速入门)
* [脚本说明](#脚本说明)
    * [脚本及样例代码](#脚本及样例代码)
    * [训练过程](#训练过程)
        * [用法](#用法)
            * [Ascend处理器环境运行](#ascend处理器环境运行)
            * [训练时推理](#训练时推理)
        * [结果](#结果)
    * [评估过程](#评估过程)
        * [用法](#用法-1)
            * [Ascend处理器环境运行](#ascend处理器环境运行-1)
        * [结果](#结果-1)
    * [导出过程](#导出过程)
        * [导出](#导出)
* [模型描述](#模型描述)
    * [性能](#性能)
        * [评估性能](#评估性能)
            * [svhn2mnist上MCD的性能](#svhn2mnist上mcd的性能)
* [随机情况说明](#随机情况说明)
* [ModelZoo主页](#modelzoo主页)
    <!-- TOC -->

# MCD描述

## 概述

MCD（Maximum Classifier Discrepancy）方法与2018年提出，旨在解决图像分类任务域适应问题。本文中作者提出了一种利用具体任务的决策边界来对源域和目标域进行数据对齐的新方法。通过将两个分类器的预测结果差异最小化，从而实现模型在目标域上的泛化能力提升。

## 论文

[Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf)

# 数据集

源域：[SVHN](http://ufldl.stanford.edu/housenumbers/)

SVHN是一个现实世界的图像数据集，用于开发机器学习和对象识别算法，对数据预处理和格式化的要求最小。它可以被视为与MNIST相似（例如，图像是小裁剪数字），但包含更多数量级的标签数据（超过60万位数字图像），并来自一个更难、未解决的现实世界问题（识别自然场景图像中的数字和数字）。SVHN是从谷歌街景图像中的门牌号中获得的。

-   10个类，每个数字1个。数字“1”有标签1，“9”有标签9，“0”有标签10。

-   73257位用于训练，26032位用于测试，531131个额外的样本，用作额外的训练数据

-   有两种格式：

    -   带有字符级边界框的原始图像。
    -   类似MNIST的32x32图像以单个字符为中心（许多图像的侧面包含一些干扰物）。

目标域：[MNIST](https://drive.google.com/file/d/1cZ4vSIS-IKoyKWPfcgxFMugw0LtMiqPf/view)

MNIST的手写数字数据库包含有有60,000个示例的训练集和10,000个示例的测试集。它是NIST提供的更大集合的子集。数字已大小规范化，并在固定大小的图像中居中。

-   数据集大小：共10个类、7万张28×28灰度图像

    -   训练集：共6万张图像
    -   测试集：共1万张图像

-   数据格式：二进制文件

-   将数据集按照如下目录形式组织

    ```shell
    $DATASET
    ├── svhn                               # svhn数据集
    │   ├── train_32x32.mat  
    │   └───test_32x32.mat
    └─── mnist_data.mat                    # mnist数据集
    ```

配置并运行`script/get_mindrecord.sh`，将数据集转换为MindRecords。scripts/build_data.sh中的参数：

```shell
--svhn_train_path                         # svhn训练数据路径
--svhn_test_path                          # svhn测试数据路径
--mnist_data_path                         # mnist数据路径
```

例如：

```shell
sh script/get_mindrecord.sh --svhn_train_path=./data/svhn/train_32x32.mat --svhn_test_path=./data/svhn/test_32x32.mat --mnist_data_path=./data/mnist_data.mat
```

# 环境要求

-   硬件（Ascend）

    -   准备Ascend处理器搭建硬件环境。
-   框架

    -   [MindSpore](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Finstall)
-   如需查看详情，请参见如下资源：

    -   [MindSpore教程](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Ftutorials%2Fzh-CN%2Fmaster%2Findex.html)
    -   [MindSpore Python API](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Fdocs%2Fzh-CN%2Fmaster%2Findex.html)
-   安装requirements.txt中的python包。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```bash
# 分布式训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]

# 分布式训练，从指定epoch开始恢复训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]

# 单机训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]

# 单机训练，从指定epoch开始恢复训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]

# 运行评估
bash scripts/run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

如果要在ModelArts上进行模型的训练，可以参考ModelArts的 [官方指导文档](https://support.huaweicloud.com/modelarts/) 开始进行模型的训练和推理，具体操作如下：

```text
# 训练模型
1. 创建作业
2. 选择数据集存储位置
3. 选择输出存储位置
2. 在模型参数列表位置按如下形式添加参数：
    data_url            [自动填充]
    train_url           [自动填充]
    checkpoint_url      [CHECKPOINT_PATH_OBS]
    modelarts           True
    run_distribute      [True/False]
    begin_epoch         [起始周期]
    eval                [True/False]
3. 选择相应数量的处理器
4. 开始运行

# 评估模型
1. 创建作业
2. 选择数据集存储位置
3. 选择输出存储位置
2. 在模型参数列表位置按如下形式添加参数：
    data_url            [自动填充]
    train_url           [自动填充]
    checkpoint_url      [CHECKPOINT_PATH_OBS]
    modelarts           True
3. 选择单个处理器
4. 开始运行
```

# 脚本说明

## 脚本及样例代码

```text
MatchNet
├── modelarts                            # ModelArts训练相关脚本
│   └── start.py                         # ModelArts训练启动脚本
├── scripts                              # Ascend执行脚本
│   ├── get_mindrecord.sh                # 自动生成MindRecord训练文件
|   ├── run_eval.sh                      # 评估脚本
│   ├── run_distribute_train.sh          # 启动Ascend910分布式训练（8卡）
│   └── run_standalone_train.sh          # 启动Ascend910单机训练（单卡）
├── src
│   ├── callback.py                      # 自定义回调函数
│   ├── customfunc.py                    # 自定义forward类
│   ├── dataset.py                       # 数据集生成器
│   ├── make_mindrecord.py               # MindRecord文件生成
│   ├── svhn2mnist.py                    # 网络定义
│   └── var_init.py                      # 网络参数初始化函数
├── eval.py                              # 910推理脚本
├── export.py                            # 模型转换脚本
├── README.md
└── train.py                             # 910训练脚本
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```bash
# 分布式训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK] 

# 分布式训练，从指定周期开始恢复训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]

# 单机训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]

# 单机训练，从指定周期开始恢复训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]
```

#### 训练时推理

如果需要训练时推理，在执行shell脚本时为 `EVAL_CALLBACK` 参数传入 `True` 即可，其默认值为 `False` 。

### 结果

使用svhn/mnist数据集训练MCD

```text
-----0.002-64-0.0001-3-----------------
epoch: 1 step: 859, loss is 1.4070675
Train epoch time: 172119.919 ms, per step time: 200.372 ms
Test set: acc1:0.6021   acc2:0.5999     acc_ensemble:0.6025     test_loss:-0.1082
epoch: 2 step: 859, loss is 1.1243069
Train epoch time: 130645.554 ms, per step time: 152.090 ms
Test set: acc1:0.6300   acc2:0.6351     acc_ensemble:0.6331     test_loss:-0.1219
epoch: 3 step: 859, loss is 1.1663135
Train epoch time: 130358.980 ms, per step time: 151.757 ms
Test set: acc1:0.6597   acc2:0.6590     acc_ensemble:0.6608     test_loss:-0.1333
...
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 运行评估
bash scripts/run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

### 结果

评估结果保存在示例路径中，文件夹名为“eval”。你可在此路径下的日志文件中找到如下结果：

```text
Number of samples: 9984, total time: 27.85 ms
============= 910 Inference =============
acc1: 0.9705 | acc2: 0.9729 | acc_ensemble: 0.9727
=========================================
```

## 导出过程

### 导出

```bash
python export.py --device_id [DEVICE_ID] --checkpoint_file [CHECKPOINT_PATH] --file_name [OUTPUT_FILENAME] --file_format [OUTPUT_FILE_FORMAT]
```

# 模型描述

## 性能

### 评估性能

#### svhn2mnist上MCD的性能

| 参数          | Ascend 910                                               |
| ------------- | -------------------------------------------------------- |
| 模型版本      | MCD                                                      |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期      | 2023-7-6                                                 |
| MindSpore版本 | 1.8.1                                                    |
| 数据集        | svhn, mnist                                              |
| 训练参数      | epoch=25, steps per epoch=859, batch_size = 64           |
| 优化器        | SGD                                                      |
| 损失函数      | SoftmaxCrossEntropyWithLogits, Discrepancy Loss, NLLLoss |
| 输出          | Accuracy                                                 |
| 损失          | 0.0629                                                   |
| 速度          | 173.718 ms/step（单卡）                                  |
| 总时长        | 45 min                                                   |

# 随机情况说明

`train.py`中使用了随机种子。

# ModelZoo主页

 请浏览官网 [主页](https://gitee.com/mindspore/models) 。