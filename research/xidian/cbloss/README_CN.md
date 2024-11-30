# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [CBLOSS描述](#CBLOSS描述)
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
            - [在Cifar-10上训练](#####在Cifar-10上训练)
        - [评估性能](#评估性能)
            - [在Cifar-10评估](#在Cifar-10上评估)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# CBLOSS描述

CBLOSS 是一种针对长尾数据分布问题的深度学习优化策略，于2019年提出，旨在解决因数据集中类别分布不均导致的模型性能下降问题。在长尾分布中，少数类别占据主导地位，而多数类别则样本稀缺，这严重影响了模型对少数类别的识别能力。CBLOSS方法通过引入一种创新的类平衡损失函数，重新定义了样本的重要性，从而提升了模型在长尾数据集上的整体性能。该方法的核心在于利用有效样本数的概念，对每个样本进行加权，以补偿因类别样本数量差异带来的训练偏差。

[论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)：Cui Y, Jia M, Lin T Y, et al. Class-balanced loss based on effective number of samples[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 9268-9277.

# 模型架构

CBLOSS方法的核心在于其独特的损失函数设计，而非特定的模型架构。它适用于各种深度学习模型，并能作为这些模型的训练损失函数进行集成。具体而言，CBLOSS的实施过程可以分为以下几个步骤：首先，对于给定的长尾数据集，计算每个类别的有效样本数，该数值通过公式（1-b^n）/（1-b）得出，其中n为样本数，b为可调节的超参数，用于控制样本重叠程度的影响。接着，基于有效样本数，设计一个重新加权方案，为不同类别的样本分配不同的权重，以在训练过程中平衡各类别的贡献。最后，将这一重新加权后的损失函数应用于深度学习模型的训练中，促使模型在长尾数据集上学习到更加均衡的特征表示和分类能力。

# 数据集

使用的数据集：[cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html)
请下载该数据集并解压放置在`.data/cifar10_dataset_directory`文件夹中，解压成功后的`.data/cifar10_dataset_directory`文件夹应包含如下文件：batches.meta.txt，data_batch_1.bin, data_batch_2.bin, data_batch_3.bin,data_batch_4, bindata_batch_5.bin, test_batch.bin。

- 数据集大小：共10个类、6万张32×32彩色图像
    - 训练集：共5万张图像
    - 测试集：共1万张图像
- 数据格式：二进制文件
    - 注：数据将在dataset.py中处理。


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
     # (4) 在网页上设置你的代码路径为 "/path/CBLOSS"
     # (5) 在网页上设置启动文件为 "train.py"
     # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
     # (7) 创建训练作业
    ```

    - 在 ModelArts 上使用单卡验证

    ```python
   # (1) 在网页上设置 "config_path='./default_config.yaml'"
   # (2) 执行a或者b
   #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
   #          在 default_config.yaml 文件中设置 其他参数
   #       b. 在网页上设置 "enable_modelarts=True"
   #          在网页上设置 其他参数
   # (3) 上传预训练模型到 S3 桶上
   # (4) 上传你的压缩数据集到 S3 桶上
   # (5) 在网页上设置你的代码路径为 "/path/CBLOSS"
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
  # (4) 在网页上设置你的代码路径为 "/path/CBLOSS"
  # (5) 在网页上设置启动文件为 "export.py"
  # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
  # (7) 创建训练作业
    ```

# 脚本说明

## 脚本及样例代码

  ```text
├── CBLOSS  
    |—— checkpoint                  # 权重文件所在文件夹
    |—— data
        |——cifar10_dataset_directory# cifar10数据集
    |── model_utils
        |——config.py                # 处理配置参数
        |——device_adapter.py        # 获取云ID
        |——local_adapter.py         # 获取本地ID
        |——moxing_adapter.py        # 参数处理
    |── models
        |──Resnet.py                # 模型结构
    |── pretrained_weights          # 预训练权重所在文件夹
    ├── scripts
        ├──run_eval.sh              # Ascend评估的shell脚本
        ├──run_eval_cpu.sh          # CPU处理器评估的shell脚本
        ├──run_eval_gpu.sh          # GPU处理器评估的shell脚本
        ├──run_train.sh             # 用于分布式训练的shell脚本
        ├──run_train_cpu.sh         # 用于CPU训练的shell脚本
        ├──run_train_gpu.sh         # 用于GPU上运行分布式训练的shell脚本
    ├── CBloss.py                   # CBLOSS文件
    ├──dataset.py                   # 处理cifar10数据集      
    ├── default_config.yaml         # 参数配置文件
    ├── eval.py                     # 评估脚本
    ├── export.py                   # 将checkpoint文件导出到air/mindir
    ├── README_CN.md                # CBLOSS相关中文说明
    ├── README.md                   # CBLOSS相关英文说明
    ├── requirements.txt            # 需要的包
    ├── train.py                    # 训练脚本
  ```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置CBLOSS。

  ```python
  device_target:'Ascend'                                   # 运行设备    
  model_root: "checkpoint"                                 # 权重文件储存文件夹  
  pretrained_model: "pretrained_weights"                   # 预训练权重文件储存文件夹 
  checkpoint_path: "model.ckpt"                            # 权重文件名称 
  epoch_num: 1000                                          # 训练epoch数  
  data_root: "data"                                        # 数据集储存文件夹
  log_step: 10                                             # 记录间隔数
  imb_ratio: 100                                           # 不平衡率
  batch_size: 50                                           # 批尺寸大小
  loss_type: 'focal'                                       # CBLOSS类型
  beta: 0.9999                                             # CBLOSS参数
  gamma: 2                                                 # CBLOSS参数
  LR: 0.01                                                 # 学习率
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
  Train Epoch: 1 [0/132 (0%)]	Loss: 0.141106
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
  Test set: Accuracy: 89%
  ```

## 导出过程

### 导出

  ```shell
  python export.py
  ```

# 模型描述

## 性能

### 训练性能

#### 在Cifar-10上训练

|         参数         |                                            Ascend                                            |
|:--------------------------:|:--------------------------------------------------------------------------------------------:|
|       模型版本        |                                             CBLOSS                                             |
|          资源          |                                  Ascend 910；CPU 24核；内存 96G;                                  |
|       上传日期        |                                          2024-11-03                                         |
|     MindSpore版本      |                                            2.2.0                                             |
|          数据集           |                                       Cifar-10                                      |
|    训练参数     |                                   num_epochs=500,batch_size=128,LR=0.01; |
|         优化器          |                                             SGD                                             |
|       损失函数        |                                          CBLOSS                                          |
|          输出           |                                              概率                                              |
|           速度            |                                  单卡：42.10毫秒/步;                                   |
|         总时长         |                                         单卡：53.46分钟;                                          |
|       参数(M)       |                                             21.31                                           |
| 微调检查点 |                                       81.2M (.ckpt文件)                                        |
|    推理模型     |                               81.2M (.onnx文件),  81.8M(.air文件)                                |

### 评估性能

#### 在Cifar-10上评估

|     参数      |           Ascend           |
|:-------------------:|:--------------------------:|
|    模型版本    |            CBLOSS            |
|      资源       | Ascend 910；CPU 24核；内存 96G; |
|    上传日期    |         2024-11-03         |
|  MindSpore 版本  |            2.2.0            |
|       数据集       |       Cifar-10, 10000张图像        |
|     batch_size      |             128             |
|       输出       |             概率             |
|      准确性       |         单卡: 89.22%         |
| 推理模型 |      81.2M (.onnx文件)       |


# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
