# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [ADDA描述](#ADDA描述)
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
            - [在MINIST和USPS上训练](#####在MINIST和USPS上训练)
        - [评估性能](#评估性能)
            - [在USPS上评估](#在USPS上评估)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# ADDA描述

ADDA 方法是2017年提出的一种用于域适应的深度学习方法，旨在解决训练数据和测试数据在分布上不同的问题。在深度学习任务中，由于不同数据集之间的分布差异，训练的模型在新的数据集上的泛化能力往往会下降。ADDA方法通过对抗训练的方式，将源域和目标域之间的差异最小化，从而实现模型在目标域上的泛化能力提升。

[论文](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf)：Tzeng E, Hoffman J, Saenko K, et al. Adversarial discriminative domain adaptation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 7167-7176.

# 模型架构

ADDA分为预训练阶段和域适应阶段。在预训练阶段，使用源域数据训练源域的特征提取器和分类器。在域适应阶段，使用目标域特征提取器和源域特征提取器分别对目标域和源域数据进行特征提取，并和判别器做对抗训练，判别器的目标是正确判断输入的特征是来自源域还是目标域，而目标域特征提取器的目标是混淆判别器，使其无法正确判别，通过这种对抗学习的方式，ADDA方法可以逐渐缩小源域特征和目标域特征的分布差异，从而提高模型在目标域上的泛化能力。最后，ADDA方法使用已经训练好的源域特征提取器和目标域分类器对目标域的数据进行分类。

# 数据集

使用的数据集：[MNIST](http://yann.lecun.com/exdb/mnist/)
请下载该数据集并解压放置在`.data/MNIST`文件夹中，解压成功后的`.data/MNIST`文件夹应包含如下文件：t10k-images-idx3-ubyte，t10k-labels-idx1-ubyte，train-images-idx3-ubyte，train-labels-idx1-ubyte。

- 数据集大小：共10个类、7万张28×28灰度图像
    - 训练集：共6万张图像
    - 测试集：共1万张图像
- 数据格式：二进制文件
    - 注：数据将在datasets/mnist.py中处理。

使用的数据集：[USPS](https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl)  
请下载该数据集并放置在`.data/USPS`文件夹中，下载成功的`.data/USPS`文件夹应包含如下文件：usps_28x28.pkl。

- 数据集大小：共10个类、9298张28×28灰度图像
    - 训练集：共7438张图像
    - 测试集：共1860张图像
- 数据格式：pkl
    - 注：数据将在datasets/usps.py中处理。

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
     # (4) 在网页上设置你的代码路径为 "/path/adda"
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
   # (5) 在网页上设置你的代码路径为 "/path/adda"
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
  # (4) 在网页上设置你的代码路径为 "/path/adda"
  # (5) 在网页上设置启动文件为 "export.py"
  # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
  # (7) 创建训练作业
    ```

# 脚本说明

## 脚本及样例代码

  ```text
├── adda  
    |—— core
        |——adapt.py                 # 目标域训练
        |——pretrain.py              # 源域训练
    |—— data
        |——MNIST                    # MNIST数据集
        |——USPS                     # USPS数据集
    |—— datasets
        |——minist.py                # 处理MNIST数据集
        |——usps.py                  # 处理USPS数据集
    |── model_utils
        |——config.py                # 处理配置参数
        |——device_adapter.py        # 获取云ID
        |——local_adapter.py         # 获取本地ID
        |——moxing_adapter.py        # 参数处理
    |── models
        |──discriminator.py         # 判别器模型结构
        |──lenet.py                 # lenet模型结构
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
    ├── README.md                   # adda相关英文说明
    ├── README_CN.md                # adda相关中文说明
    ├── requirements.txt            # 需要的包
    ├── train.py                    # 训练脚本
    ├── uilts.py                    # 工具文件
  ```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置adda。

  ```python
  device_target:'Ascend'                                   # 运行设备  
  dataset_mean: 0.5                                        # 归一化均值  
  dataset_std: 0.5                                         # 归一化标准差  
  batch_size:50                                            # 训练批次大小  
  src_dataset: "MNIST"                                     # 源域数据集  
  src_encoder_checkpoint: "ADDA-source-encoder.ckpt"       # 源域编码器权重文件  
  src_classifier_checkpoint: "ADDA-source-classifier.ckpt" # 源域分类器权重文件  
  tgt_dataset: "USPS"                                      # 目标域数据集  
  tgt_encoder_checkpoint: "ADDA-target-encoder.ckpt"       # 目标域编码器权重文件  
  model_root: "checkpoint"                                 # 权重文件储存文件夹  
  d_input_dims: 500                                        # 判别器输入层维度  
  d_hidden_dims: 500                                       # 判别器隐藏层维度  
  d_output_dims: 2                                         # 判别器输出层维度  
  d_model_checkpoint: "ADDA-critic.ckpt"                   # 判别器权重文件  
  num_epochs_pre: 10                                       # 源域训练epoch数  
  log_step_pre: 600                                        # 源域记录间隔  
  eval_step_pre: 2                                         # 源域测试间隔  
  save_step_pre: 2                                         # 源域保存间隔  
  num_epochs: 60                                           # 目标域训练epoch数  
  save_step: 10                                            # 目标域保存间隔  
  d_learning_rate: 3.0e-4                                  # 编码器学习率  
  c_learning_rate: 1.0e-4                                  # 判别器学习率  
  beta1: 0.5                                               # 第一个动量矩阵的指数衰减率  
  beta2: 0.9                                               # 第二个动量矩阵的指数衰减率  
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
  Avg Accuracy = 97.027027%
  ```

## 导出过程

### 导出

  ```shell
  python export.py
  ```

# 模型描述

## 性能

### 训练性能

#### 在MINIST和USPS上训练

|         参数         |                                            Ascend                                            |
|:--------------------------:|:--------------------------------------------------------------------------------------------:|
|       模型版本        |                                             ADDA                                             |
|          资源          |                                  Ascend 910；CPU 24核；内存 96G;                                  |
|       上传日期        |                                          2023-05-27                                          |
|     MindSpore版本      |                                            1.7.1                                             |
|          数据集           |                                      源域：MNIST,目标域：USPS;                                      |
|    训练参数     | num_epochs_pre=10,num_epochs=60,batch_size=50,d_learning_rate=3.0e-4,c_learning_rate=1.0e-4; |
|         优化器          |                                             Adam                                             |
|       损失函数        |                                          Softmax交叉熵                                          |
|          输出           |                                              概率                                              |
|           速度            |                                  单卡：7毫秒/步（源域），79毫秒/步（目标域）;                                   |
|         总时长         |                                         单卡：22.78分钟;                                          |
|       参数(M)       |                                             0.431                                             |
| 微调检查点 |                                       1.63M (.ckpt文件)                                        |
|    推理模型     |                               1.64M (.onnx文件),  1.65M(.air文件)                                |

### 评估性能

#### 在USPS上评估

|     参数      |           Ascend           |
|:-------------------:|:--------------------------:|
|    模型版本    |            ADDA            |
|      资源       | Ascend 910；CPU 24核；内存 96G; |
|    上传日期    |         2023-05-27         |
|  MindSpore 版本  |           1.7.1            |
|       数据集       |       USPS, 1860张图像        |
|     batch_size      |             50             |
|       输出       |             概率             |
|      准确性       |         单卡: 97.03%         |
| 推理模型 |      1.64M (.onnx文件)       |

# 随机情况说明

在train.py中，我们使用utiles.py中的init_random_seed()函数设置了随机数种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
