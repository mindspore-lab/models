# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [RelationNet描述](#RelationNet描述)
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
        - [评估性能](#评估性能)

- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# RelationNet描述
RelationNet是一种用于解决关系推理问题的神经网络模型，尤其适用于需要理解对象之间相互作用的任务，如图像中的物体关系识别。它由Facebook人工智能研究实验室（FAIR）的研究人员在2017年提出。

RelationNet设计的核心思想是让机器学习系统能够像人类一样理解和处理图像中不同部分之间的关系。传统的方法通常依赖于特征的直接比较或者预先定义的关系模型，而RelationNet则通过引入一种新的机制来学习这些关系，这种方法使得模型可以在没有显式编程的情况下理解复杂的场景。


RelationNet的一个关键特性是它可以应用于不同的任务上，比如图像问答（Visual Question Answering, VQA）、物体识别和关系推理等。通过训练，RelationNet可以学会如何识别图像中的物体以及它们之间的相对位置和其他关系，从而更好地回答关于图像的问题。

总之，RelationNet提供了一种有效的方式，使深度学习模型能够处理更加复杂的关系推理任务，这在视觉理解领域是一个重要的进步。随着研究的发展，这样的模型有望进一步推动计算机视觉和自然语言处理等领域的发展。

[论文](https://arxiv.org/abs/1711.06025)：Sung F, Yang Y, Zhang L, et al. Learning to compare: Relation network for few-shot learning[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 1199-1208.

# 模型架构

该模型由两个主要部分组成：

- 关系检测器（Relation Detector）：这部分负责从输入数据中提取出可能的关系对，并为每个关系对生成一个向量表示。
- 关系分类器（Relation Classifier）：这部分接收关系检测器产生的向量，并预测这些关系的重要性或相关性。

# 数据集

使用的数据集：[Omniglot](https://gitcode.net/mirrors/brendenlake/omniglot?utm_source=csdn_github_accelerator)
请下载该数据集并解压放置在`.data/`文件夹中，解压成功后的`.data/omniglot_resized`文件夹用于训练和测试

 Omniglot数据集由50种字母表（每种字母表的字符数不同），每种字母表包括不同字符，比如常见的Latin拉丁字母表即abcdefg，共26个字母，还有韩语，日语，共1623种字符，每个字符又是有20个人不同的写法，每个写法是一张108*108的图像，即该数据集的大小是1623*20


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
  python omniglot_train_few_shot.py > mniglot_train_few_shot.log 2>&1 &

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
  python omniglot_train_few_shot.py > omniglot_train_few_shot.log 2>&1 &

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
  python omniglot_train_few_shot.py > omniglot_train_few_shot.log 2>&1 &

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
     # (4) 在网页上设置你的代码路径为 "/path/RelationNet"
     # (5) 在网页上设置启动文件为 "omniglot_train_few_shot.py"
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
   # (5) 在网页上设置你的代码路径为 "/path/RelationNet"
   # (6) 在网页上设置启动文件为 "eval.py"
   # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
   # (8) 创建训练作业
    ```  

    - 在 ModelArts 上使用单卡导出模型

    ```python
  # (1) 在网页上设置 "config_path='./default_config.yaml'"
  # (2) 在 default_config.yaml 文件中设置 "enable_modelarts=True"
  # (3) 上传你的预训练模型到 S3 桶上
  # (4) 在网页上设置你的代码路径为 "/path/RelationNet"
  # (5) 在网页上设置启动文件为 "export.py"
  # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
  # (7) 创建训练作业
    ```

# 脚本说明

## 脚本及样例代码

```text
├── RelationNet  
  ├── README_CN.md                  # RelationNet相关中文说明
  ├── README.md                     # RelationNet相关英文说明
  ├── checkpoints
  │   ├── omniglot_feature_encoder.ckpt
  │   └── omniglot_relation_network.ckpt
  ├── data
  │   ├── omniglot_resized          # 数据集
  │   └── __MACOSX
  ├── default_config.yaml           # 参数配置文件
  ├── eval.py                       # 评估脚本
  ├── export.py                     # 将checkpoint文件导出到air/mindir
  ├── model_utils
  │   ├── config.py                 # 处理配置参数
  │   ├── device_adapter.py         # 获取云ID
  │   ├── local_adapter.py          # 获取本地ID
  │   ├── moxing_adapter.py         # 参数处理
  │   └── utils.py                  # 工具文件
  ├── models
  │   └── models.py
  ├── omniglot_train_few_shot.py    # 训练脚本
  ├── scripts
  │   ├── run_eval_cpu.sh           # CPU处理器评估的shell脚本
  │   ├── run_eval_gpu.sh           # GPU处理器评估的shell脚本
  │   ├── run_eval.sh               # Ascend评估的shell脚本
  │   ├── run_train_cpu.sh          # 用于CPU训练的shell脚本
  │   ├── run_train_gpu.sh          # 用于GPU上运行分布式训练的shell脚本
  │   └── run_train.sh              # 用于分布式训练的shell脚本 
  ├── requirements.txt              # 需要的包
  └── task_generator.py             # 数据集预处理

```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- RelationNet

  ```python

  enable_modelarts: False
  device_target: "Ascend" # Ascend                              # 运行设备  
  DEVICE_ID: 3                                                  # 设备编号 

  model_root: "checkpoints"
  encoder_checkpoint: "omniglot_feature_encoder.ckpt"
  relation_checkpoint: "omniglot_relation_network.ckpt"

  CLASS_NUM: 5                                                  # 类别数目
  SAMPLE_NUM_PER_CLASS: 1                                       # 每类样本的数目
  FEATURE_DIM: 64                                               # 特征维度
  RELATION_DIM: 8                                               # 关系维度
  BATCH_NUM_PER_CLASS: 19                                       # 每批样本数目
  EPISODE: 300000                                               # 训练批数
  TEST_EPISODE: 100                                             # 测试批数
  LEARNING_RATE: 0.001                                          # 学习率
  PRINT_FREQUENCY: 100                                          # 打印频率
  TEST_FREQUENCY: 300                                           # 测试频率

  file_name: "net"                                              # 导出文件名称  
  file_format: "MINDIR"                                         # 导出文件格式  
  image_height: 28                                              # 样本数据图像的高度 
  image_width: 28                                               # 样本数据图像的宽度
  ```

更多配置细节请参考脚本`config.py`。  

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python omniglot_train_few_shot.py > omniglot_train_few_shot.log 2>&1 &
  ```

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式获得损失值：

  ```bash
  # grep "loss" train.log
  Training...
  episode:100    loss:0.10610510    Time of train one episode:62.903ms
  episode:200    loss:0.07776772    Time of train one episode:62.420ms
  episode:300    loss:0.06913537    Time of train one episode:67.289ms

  ```

- GPU处理器环境运行
     请将配置文件default_config.yaml中的`device_target`从`Ascend`改为`GPU`

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python omniglot_train_few_shot.py > omniglot_train_few_shot.log 2>&1 &
  或
  bash run_train_gpu.sh 8 0,1,2,3,4,5,6,7  
  ```

- CPU处理器环境运行
     请将配置文件default_config.yaml中的`device_target`从`Ascend`改为`CPU`

  ```bash
  python omniglot_train_few_shot.py  > omniglot_train_few_shot.log 2>&1 &
  或
  bash run_train_cpu.sh
  ```

  上述所有shell命令将在后台运行，您可以通过`train/omniglot_train_few_shot.log`文件查看结果。

### 分布式训练

- GPU处理器环境运行
  请将配置文件default_config.yaml中的`device_target`从`Ascend`改为`GPU`

  ```bash
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7
  ```

  上述shell命令将在后台运行分布训练。您可以通过`train/omniglot_train_few_shot.log`文件查看结果。

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
  Avg Accuracy = 99.4%
  ```

## 导出过程

### 导出

  ```shell
  python export.py
  ```

# 模型描述

## 性能

### 训练性能

|         参数         |                                            Ascend                                            |
|:--------------------------:|:--------------------------------------------------------------------------------------------:|
|       模型版本        |                                             RelatinNet                                             |
|          资源          |                                  Ascend 910；CPU 24核；内存 96G;                                  |
|       上传日期        |                                          2024-09-07                                          |
|     MindSpore版本      |                                            2.2.0                                             |
|          数据集           |                                      Omniglot;                                      |
|    训练参数     |                       sample_per_class=1;batch_per_class=19;class_num=5;lr=1e-3;                                        |
|         优化器          |                                             Adam                                             |
|       损失函数        |                                          MSE损失                                          |
|          输出           |                                              概率                                              |
|           速度            |                                 单卡：62.903毫秒/代;                                   |
|         总时长         |                                         单卡：182.45分钟;                                          |
|       参数(M)       |                                             223k                                             |
| 微调检查点 |                                       0.887M (.ckpt文件)                                        |
|    推理模型     |                               0.901M (.onnx文件),  0.21M(.air文件)                                |

### 评估性能

|     参数      |           Ascend           |
|:-------------------:|:--------------------------:|
|    模型版本    |            RelatinNet            |
|      资源       | Ascend 910；CPU 24核；内存 96G; |
|    上传日期    |         2024-09-07         |
|  MindSpore 版本  |           2.2.0            |
|       数据集       |       Omniglot        |
|       输出       |             概率             |
|      准确性       |         单卡: 99.4%         |
| 推理模型 |      0.901M (.onnx文件)       |



# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。

