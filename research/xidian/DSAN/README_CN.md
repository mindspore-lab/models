# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [DSAN描述](#DSAN描述)
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
            - [在OFFICE-31上训练](#####在OFFICE-31上训练)
        - [评估性能](#评估性能)
            - [在OFFICE-31上评估](#在OFFICE-31上评估)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# DSAN描述

DSAN是一种针对无监督域适应任务设计的深度学习方法，旨在解决传统域适应方法存在的全局域偏移问题。传统的域适应方法往往只关注全局源域和目标域之间的分布对齐，而忽略了同一类别内不同子域之间的关系，这导致了迁移学习性能的不足。DSAN则通过关注子域自适应，精确对齐相关子域的分布，从而在不损失细粒度信息的情况下提升迁移学习的效果。该方法基于局部最大均值差异（LMMD）来对齐不同域上域特定层激活的相关子域分布，实现传输网络的学习。

[论文](https://ieeexplore.ieee.org/abstract/document/9085896)：Zhu Y, Zhuang F, Wang J, et al. Deep subdomain adaptation network for image classification[J]. IEEE transactions on neural networks and learning systems, 2020, 32(4): 1713-1722.

# 模型架构

DSAN模型架构的核心在于其独特的子域对齐机制，无需对抗训练，收敛速度快。具体来说，DSAN模型可以分为特征提取层、子域对齐层和分类层。在特征提取层，DSAN利用深度神经网络从源域和目标域中提取特征表示。接着，在子域对齐层，DSAN通过计算局部最大均值差异（LMMD）损失来对齐不同域上域特定层激活的相关子域分布。LMMD损失的设计使得DSAN能够精确捕捉并对齐相关子域的分布，从而提高了迁移学习的性能。最后，在分类层，DSAN使用已经对齐的特征表示对目标域的数据进行分类。

# 数据集

使用的数据集：[Office]https://faculty.cc.gatech.edu/~judy/domainadapt/)
请下载该数据集并解压放置在`.data/OFFICE31`文件夹中，解压成功后的`.data/OFFICE31`文件夹应包含如下文件夹：amazon，dslr，webcam，三者分别代表不同的域。

- amazon数据集大小：共31个类、2,817张224×224彩色图像
- dslr数据集大小：共31个类、498张224×224彩色图像
- webcam数据集大小：共31个类、795张224×224彩色图像
- 数据格式：图像文件
- 注：数据将在data_loader.py中处理。
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
在训练和评估之前，可以在default_config_cpu.yaml文件中指定源域数据集和目标域数据，即设置“ src_dataset：XX”，“ tgt_dataset：XX ”，其中’XX‘可选择'MNIST'或'USPU'。  

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
     # (4) 在网页上设置你的代码路径为 "/path/DSAN"
     # (5) 在网页上设置启动文件为 "train.py"
     # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
     # (7) 创建训练作业
    ```

    - 在 ModelArts 上使用单卡验证 USPS 数据集

    ```python
   # (1) 在网页上设置 "config_path='./default_config.yaml'"
   # (2) 执行a或者b
   #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
   #          在 default_config.yaml 文件中设置 其他参数
   #       b. 在网页上设置 "enable_modelarts=True"
   #          在网页上设置 其他参数
   # (3) 上传预训练模型到 S3 桶上
   # (4) 上传你的压缩数据集到 S3 桶上
   # (5) 在网页上设置你的代码路径为 "/path/DSAN"
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
  # (4) 在网页上设置你的代码路径为 "/path/DSAN"
  # (5) 在网页上设置启动文件为 "export.py"
  # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
  # (7) 创建训练作业
    ```

# 脚本说明

## 脚本及样例代码

  ```text
├── DSAN  
    |—— checkpoint                  # 权重文件所在文件夹     
    |—— data         
        |——OFFICE31                 # OFFICE31数据集
    |── model_utils
        |——config.py                # 处理配置参数
        |——device_adapter.py        # 获取云ID
        |——local_adapter.py         # 获取本地ID
        |——moxing_adapter.py        # 参数处理
    |── models
        |──LoadPretrainedModel      # 预训练权重
        |──DSAN.py                  # DSAN模型结构
        |──RESNET.py                # resnet模型结构
    ├── scripts
        ├──run_eval.sh              # Ascend评估的shell脚本
        ├──run_eval_cpu.sh          # CPU处理器评估的shell脚本
        ├──run_eval_gpu.sh          # GPU处理器评估的shell脚本
        ├──run_train.sh             # 用于分布式训练的shell脚本
        ├──run_train_cpu.sh         # 用于CPU训练的shell脚本
        ├──run_train_gpu.sh         # 用于GPU上运行分布式训练的shell脚本
    ├── default_config.yaml          # 参数配置文件
    ├── eval.py                     # 评估脚本
    ├── export.py                   # 将checkpoint文件导出到air/mindir
    ├── README.md                   # DSAN相关英文说明
    ├── README_CN.md                # DSAN相关中文说明
    ├── requirements.txt            # 需要的包
    ├── train.py                    # 训练脚本
  ```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置DSAN。

  ```python
  # params for train
  device_target:'Ascend'                                   # 运行设备  
  nepoch: 200                                              # 训练epoch数  
  lr: [0.001, 0.01, 0.01]                                  # 学习率
  seed: 2021                                               # 随机种子
  weight: 0.5                                              # lmmd损失权重
  momentum: 0.9                                            # 动量项
  decay: 5e-4                                              # 衰减率
  bottleneck: True                                         # 是否添加瓶颈层
  log_interval: 10                                         # 记录间隔

  # params for dataset
  nclass: 31                                               # 类别数目
  batch_size: 32                                           # 训练批次大小  
  src: 'amazon'                                            # 源域数据集 
  tar: 'webcam'                                            # 目标域数据集  
  model_root: "checkpoint"                                 # 权重文件储存文件夹  
  dataset_path: 'data/OFFICE31'                            # 数据集储存文件夹
  file_name: "net"                                         # 导出文件名称  
  file_format: "ONNX"                                      # 导出文件格式  
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
  Epoch:1,Step:7,Loss_lmmd:3.3463
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
  Avg Accuracy = 90.078704% 
  ```

## 导出过程

### 导出

  ```shell
  python export.py
  ```

# 模型描述

## 性能

### 训练性能
 
#### 在OFFICE-31上训练

|         参数         |                                            Ascend                                            |
|:--------------------------:|:--------------------------------------------------------------------------------------------:|
|       模型版本        |                                             DSAN                                             |
|          资源          |                                  Ascend 910；CPU 24核；内存 96G;                                  |
|       上传日期        |                                          2024-11-06                                         |
|     MindSpore版本      |                                            2.2.0                                             |
|          数据集           |                                      源域：amazon,目标域：webcam;                                      |
|    训练参数     | nepoch=200,lr=[0.001, 0.01, 0.01],weight=0.5,momentum=0.9,decay=5e-4,batch_size=32; |
|         优化器          |                                             SGD                                             |
|       损失函数        |                                          lmmd,交叉熵                                          |
|          输出           |                                              概率                                              |
|           速度            |                                  单卡：1631毫秒/步;                                   |
|         总时长         |                                         单卡：80.48分钟;                                          |
|       参数(M)       |                                             24.09                                             |
| 微调检查点 |                                       91.9M (.ckpt文件)                                        |
|    推理模型     |                               91.9M (.onnx文件),  91.9M(.air文件)                                |

### 评估性能

#### 在OFFICE-31上评估

|     参数      |           Ascend           |
|:-------------------:|:--------------------------:|
|    模型版本    |            DSAN            |
|      资源       | Ascend 910；CPU 24核；内存 96G; |
|    上传日期    |         2024-11-06         |
|  MindSpore 版本  |           2.2.0            |
|       数据集       |       webcam, 795张图像        |
|     batch_size      |             32             |
|       输出       |             概率             |
|      准确性       |         单卡: 94.70%         |
| 推理模型 |      91.9M (.onnx文件)       |

# 随机情况说明

在train.py中，我们使用train.py中的np.random.seed(SEED)设置了随机数种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
