# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [FADA描述](#FADA描述)
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
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [在MINIST和SVHN上训练](#####在MINIST和SVHN上训练)
        - [评估性能](#评估性能)
            - [在SVHN上评估](#在SVHN上评估)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# FADA描述

FADA 方法是2017年提出的一种用于域适应的深度学习方法。这篇论文的应用场景是只有很少的有标注目标域数据可用。为了解决目标域训练样本缺乏，作者将这些样本与每个源域训练样本构造成pairs，可分成四组：
- 1、组成一对的样本都来自源域，且标签相同；
- 2、组成一对的样本分别来自源域和目标域，标签相同；
- 3、组成一对的样本都来自源域，标签不同；
- 4、组成一对的样本分别来自源域和目标域，标签不同；

[论文](https://papers.nips.cc/paper_files/paper/2017/hash/21c5bba1dd6aed9ab48c2b34c1a0adde-Abstract.html)：Motiian S, Jones Q, Iranmanesh S, et al. Few-shot adversarial domain adaptation[J]. Advances in neural information processing systems, 2017, 30.

# 模型架构

FADA 共分为三步。第一步用源域预训练，初始化g和h，最小化分类损失。第二步是冻结g，训练一个domain-class discriminator (DCD)；前文提到的四分类，最大程度混淆这四组样本对的分类。第三步是固定DCD，更新g和h。最后，FADA方法使用已经训练好的g和g对目标域的数据进行分类。

# 数据集

使用的数据集：[MNIST](http://yann.lecun.com/exdb/mnist/)
请下载该数据集并解压放置在`.dataset/MNIST_Data`文件夹中，解压成功后的`.data/MNIST_Data`文件夹应包含如下文件：t10k-images-idx3-ubyte，t10k-labels-idx1-ubyte，train-images-idx3-ubyte，train-labels-idx1-ubyte。

- 数据集大小：共10个类、7万张28×28灰度图像
    - 训练集：共6万张图像
    - 测试集：共1万张图像
- 数据格式：二进制文件
    - 注：数据将在model/dataloader.py中处理。

使用的数据集：[SVHN](http://ufldl.stanford.edu/housenumbers/)  
请下载该数据集并放置在`.dataset/SVHN`文件夹中，下载成功的`.data/SVHN`文件夹应包含如下文件：test_32x32.mat,train_32x32.mat。

- 数据集大小：共10个类、99289张28×28灰度图像
    - 训练集：共73257张图像
    - 测试集：共26032张图像
- 数据格式：mat
    - 注：数据将在model/dataloader.py中处理。

# 特性

## 混合精度

采用混合精度的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

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
  python train.py > train.log 2>&1 &

  # 运行分布式训练示例
  bash run_train.sh [RANK_TABLE_FILE]
  # example: bash run_train.sh ~/hccl_8p.json

  # 运行评估示例
  python eval.py > eval.log 2>&1 &
  或
  bash run_eval.sh
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：
<https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- GPU处理器环境运行

  为了在GPU处理器环境运行，请将配置文件default_config.yaml中的`device_target`从`Ascend`改为`GPU`

  ```python
  # 运行训练示例
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &

  # 运行分布式训练示例
  bash run_train_gpu.sh 8 0,1,2,3,4,5,6,7

  # 运行评估示例
  python eval.py > eval.log 2>&1 &
  或
  bash run_eval_gpu.sh
  ```

- CPU处理器环境运行

  为了在CPU处理器环境运行，请将配置文件default_config.yaml中的`device_target`从`Ascend`改为CPU

  ```python
  # 运行训练示例
  bash run_train_cpu.sh
  或
  python train.py > train.log 2>&1 &

  # 运行评估示例
  bash run_eval_cpu.sh
  或
  python eval.py > eval.log 2>&1 &
  ```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))  

    - 在 ModelArts 上使用8卡训练

    ```python
     # (1) 在网页上设置 "config_path='./default_config.yaml'"
     # (2) 执行a或者b
     #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
     #          在 default_config.yaml 文件中设置 其他参数
     #       b. 在网页上设置 "enable_modelarts=True"
     #          在网页上设置 其他参数
     # (3) 上传压缩数据集到 S3 桶上
     # (4) 在网页上设置你的代码路径为 "/path/FADA"
     # (5) 在网页上设置启动文件为 "train.py"
     # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
     # (7) 创建训练作业
    ```

    - 在 ModelArts 上使用单卡验证 SVHN 数据集

    ```python
   # (1) 在网页上设置 "config_path='./default_config.yaml'"
   # (2) 执行a或者b
   #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
   #          在 default_config.yaml 文件中设置 其他参数
   #       b. 在网页上设置 "enable_modelarts=True"
   #          在网页上设置 其他参数
   # (3) 上传预训练模型到 S3 桶上
   # (4) 上传你的压缩数据集到 S3 桶上
   # (5) 在网页上设置你的代码路径为 "/path/FADA"
   # (6) 在网页上设置启动文件为 "eval.py"
   # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
   # (8) 创建训练作业
    ```  

    - 在 ModelArts 上使用单卡导出模型

    ```python
  # (1) 在网页上设置 "config_path='./default_config.yaml'"
  # (2) 执行a或者b
  #       a. 在 cifar10_config.yaml 文件中设置 "enable_modelarts=True"
  #          在 cifar10_config.yaml 文件中设置 其他参数
  #       b. 在网页上设置 "enable_modelarts=True"
  #          在网页上设置 其他参数
  # (3) 上传你的预训练模型到 S3 桶上
  # (4) 在网页上设置你的代码路径为 "/path/FADA"
  # (5) 在网页上设置启动文件为 "export.py"
  # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
  # (7) 创建训练作业
    ```

# 脚本说明

## 脚本及样例代码

  ```text

  ```text
├── FADA  
  ├── checkpoint
  ├── core
  │   ├── step1.py              # 第一步训练
  │   ├── step2.py              # 第二步训练
  │   └── step3.py              # 第三步训练
  ├── dataset
  │   ├── MNIST_Data            # MNIST数据集
  │   └── SVHN                  # SVHN数据集
  ├── model
  │   ├── dataloader.py         # 处理数据集，构建Dataloader
  │   ├── model.py              # 模型结构
  │   └── utils.py              # 工具文件
  ├── model_utils
  │   ├── config.py             # 处理配置参数
  │   ├── device_adapter.py     # 获取云ID
  │   ├── local_adapter.py      # 获取本地ID
  │   ├── moxing_adapter.py     # 参数处理
  │   └── utils.py              # 工具文件
  ├── scripts
  │   ├── run_eval_cpu.sh       # CPU处理器评估的shell脚本
  │   ├── run_eval_gpu.sh       # GPU处理器评估的shell脚本
  │   ├── run_eval.sh           # Ascend评估的shell脚本
  │   ├── run_train_cpu.sh      # 用于CPU训练的shell脚本
  │   ├── run_train_gpu.sh      # 用于GPU上运行分布式训练的shell脚本
  │   └── run_train.sh          # 用于分布式训练的shell脚本 
  ├── README_CN.md              # FADA相关中文说明
  ├── README.md                 # FADA相关英文说明
  ├── requirements.txt          # 需要的包
  ├── default_config.yaml       # 参数配置文件
  ├── eval.py                   # 评估脚本
  ├── export.py                 # 将checkpoint文件导出到air/mindir
  └── train.py                  # 训练脚本
  ```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置FADA。

  ```python
  enable_modelarts: False 
  device_target: "Ascend"                                         # 运行设备  
  model_root: "checkpoint"
  src_encoder_checkpoint: "FADA-source-encoder.ckpt"              # 源域编码器权重文件  
  src_classifier_checkpoint: "FADA-source-classifier.ckpt"        # 源域分类器权重文件  
  tgt_discriminator_checkpoint: "FADA-tgt-discriminator.ckpt"     # 目标域鉴别器权重文件
  tgt_encoder_checkpoint: "FADA-tgt-encoder.ckpt"                 # 目标域编码器权重文件  
  tgt_classifier_checkpoint: "FADA-tgt-classifier.ckpt"           # 目标域分类器权重文件  
  n_epoch_1: 10                                                   # 第一步训练epoch数  
  n_epoch_2: 100                                                  # 第二步训练epoch数  
  n_epoch_3: 100                                                  # 第三步训练epoch数  
  disc_feature: 128                                               # 鉴别器输入特征的大小
  src_lr: 1e-3                                                    # 第一步源域训练的学习率
  dcd_lr_2: 1e-3                                                  # 第二步训练鉴别器的学习率
  CE_lr_3: 3e-3                                                   # 第三步训练编码器和分类器的学习率
  D_lr_3: 3e-3                                                    # 第三步训练鉴别器的学习率  
  loss_dcd_weight: 0.2                                            # 鉴别器损失的权重
  n_target_samples: 7                                             # 样本配对的数目
  batch_size: 64                                                  # 训练批次大小  
  mini_batch_size_g_h: 20                                         # 训练编码器和分类器的最小批次大小
  mini_batch_size_dcd: 40                                         # 训练鉴别器的最小批次大小
  file_name: "net"                                                # 导出文件名称  
  file_format: "MINDIR"                                           # 导出文件格式  
  image_height: 28                                                # 样本数据图像的高度 
  image_width: 28                                                 # 样本数据图像的宽度
  ```

更多配置细节请参考脚本`config.py`。  

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py > train.log 2>&1 &
  ```

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式获得损失值：

  ```bash
  # grep "loss is " train.log
  Epoch [1/10] Step [600]: loss=0.14988492  Epoch [1/10] Step [1200]: loss=0.124895595
  ```

- GPU处理器环境运行
     请将配置文件default_config.yaml中的`device_target`从`Ascend`改为`GPU`

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &
  或
  bash run_train_gpu.sh 8 0,1,2,3,4,5,6,7  
  ```

- CPU处理器环境运行
     请将配置文件default_config.yaml中的`device_target`从`Ascend`改为`CPU`

  ```bash
  python train.py  > train.log 2>&1 &
  或
  bash run_train_cpu.sh
  ```

  上述所有shell命令将在后台运行，您可以通过`train/train.log`文件查看结果。

### 分布式训练

- GPU处理器环境运行
  请将配置文件default_config.yaml中的`device_target`从`Ascend`改为`GPU`

  ```bash
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7
  ```

  上述shell命令将在后台运行分布训练。您可以通过`train/train.log`文件查看结果。

## 评估过程

### 评估  

- 在Ascend环境运行评估目标域数据集

  ```bash
  python eval.py > eval.log 2>&1 &
  或
  bash run_eval.sh
  ```

- 在GPU处理器环境运行时评估目标域数据集

  请将配置文件default_config.yaml中的`device_target`从`Ascend`改为`GPU`

  ```bash
  python eval.py > eval.log 2>&1 &
  或
  bash run_eval_gpu.sh
  ```

- 在CPU处理器环境运行时评估目标域数据集

  请将配置文件default_config.yaml中的`device_target`从`Ascend`改为`CPU`

  ```bash
  python eval.py > eval.log 2>&1 &
  或
  bash run_eval_cpu.sh
  ```

  上述所有命令将在后台运行，您可以通过`eval/eval.log`文件查看结果。测试数据集的准确性如下：

  ```bash
  Avg Accuracy = 47.027027%
  ```

## 导出过程

### 导出

  ```shell
  python export.py
  ```

# 模型描述

## 性能

### 训练性能

#### 在MINIST和SVHN上训练

|         参数         |                                            Ascend                                            |
|:--------------------------:|:--------------------------------------------------------------------------------------------:|
|       模型版本        |                                             FADA                                             |
|          资源          |                                  Ascend 910；CPU 24核；内存 96G;                                  |
|       上传日期        |                                          2024-01-02                                          |
|     MindSpore版本      |                                            1.10.1                                             |
|          数据集           |                                      源域：MNIST,目标域：SVHN;                                      |
|    训练参数     | n_epoch_1=10,n_epoch_2=60,n_epoch_3=60,batch_size=64,src_lr=1e-3,dcd_lr_2=1.0e-3; |
|         优化器          |                                             Adam                                             |
|       损失函数        |                                          Softmax交叉熵                                          |
|          输出           |                                              概率                                              |
|           速度            |                                 单卡：6.176毫秒/步（step1），0.142毫秒/步（step2），446.702毫秒/步（step2）;                                   |
|         总时长         |                                         单卡：182.45分钟;                                          |
|       参数(M)       |                                             49.444k                                             |
| 微调检查点 |                                       0.19M (.ckpt文件)                                        |
|    推理模型     |                               0.19M (.onnx文件),  0.21M(.air文件)                                |

### 评估性能

#### 在SVHN上评估

|     参数      |           Ascend           |
|:-------------------:|:--------------------------:|
|    模型版本    |            FADA            |
|      资源       | Ascend 910；CPU 24核；内存 96G; |
|    上传日期    |         2024-01-02         |
|  MindSpore 版本  |           1.10.1            |
|       数据集       |       SVHN, 26032张图像        |
|     batch_size      |             64             |
|       输出       |             概率             |
|      准确性       |         单卡: 47.15%         |
| 推理模型 |      0.19M (.onnx文件)       |

# 随机情况说明

在train.py中，我们使用utiles.py中的init_random_seed()函数设置了随机数种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。

