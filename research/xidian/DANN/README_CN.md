# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [DANN描述](#Hardnet描述)
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
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [ModelArts上训练DANN](#modelArts上训练Hardnet)
        - [评估性能](#评估性能)
            - [ModelArts上评估DANN](#liberty上评估Hardnet)
    - [使用流程](#使用流程)
        - [推理](#推理-1)
        - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# DANN描述

DANN是一个经典的域适应网络框架，论文作者Yaroslav Ganin等人首次将对抗的思想引入迁移学习领域当中，简单却很有效。作者在多个数据集上进行实验测试，均取得了state-of-the-art的结果。

[论文](https://arxiv.org/abs/1409.7495)：Ganin Y , Lempitsky V . Unsupervised Domain Adaptation by Backpropagation[C]// JMLR.org. JMLR.org, 2014.
# 模型架构
DANN的骨干网络与LeNet类似，含有卷积层以及最大池化层，卷积层的步幅为1，不断增加特征的通道数量；最大池化层用于实现下采样。在每个卷积层后使用批量归一化，每个池化层后使用ReLU激活函数。最后输出50维的描述子。
DANN的类别分类器与域分类器结构相似，都是全连接层后使用批量归一化和ReLU激活函数，其中第一层全连接需要使用Dropout层防止网络过拟合。

# 数据集

使用的数据集：[MNIST](<http://yann.lecun.com/exdb/mnist/>)
[MNIST_M](<http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz>)

- MNIST数据集介绍：该数据集是黑底白字的手写体数字图片，中每张图片是一个28*28像素点的0 ~ 9的灰质手写数字图片，图像像素值为0~255；图片是以字节形式存储的3维数据。
    - 训练与测试：训练集train一共包含了60000张图像和标签，而测试集一共包含了10000张图像和标签
- 
- MNIST_M数据集介绍：该数据集是由MNIST数字与BSDS500数据集中的随机色块混合而成

    - 注：数据将在src/data_loader.py中处理。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
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

  为了在GPU处理器环境运行，请将配置文件config.yaml中的`device_target`从`Ascend`改为`GPU`  
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
  
默认使用MNIST数据集作为源域，MNIST_M数据集作为目标域。如需查看更多详情，请参考指定脚本。

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用8卡训练 MNIST和mnist_m 数据集

      ```python
      # (1) 上传你的解压数据集到 S3 桶上
      # (2) 在网页上设置你的代码路径为 "/DANN"
      # (3) 在网页上设置启动文件为 "train.py"
      # (4) 在网页上设置"数据集位置"、"训练输出文件路径"、"作业日志路径"等
      # (5) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡验证 MNIST和mnist_m 数据集

      ```python
      # (1) 上传你的解压数据集到 S3 桶上
      # (2) 在网页上设置你的代码路径为 "/DANN"
      # (3) 在网页上设置启动文件为 "eval.py"
      # (4) 在网页上设置"数据集位置"、"作业日志路径"等
      # (5) 创建训练作业
      ```
    

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── DANN
        ├── README_CN.md                // DANN相关中文说明
        ├── README.md                   // DANN相关英文说明
        ├── model_utils
            ├── config.py               // 配置文件
        ├── models
        ├── scripts
        │   ├──run_train.sh             // 分布式到Ascend的shell脚本
        │   ├──run_train_gpu.sh         // 分布式到GPU处理器的shell脚本
        │   ├──run_train_cpu.sh         // CPU处理器训练的shell脚本
        │   ├──run_eval.sh              // Ascend评估的shell脚本
        │   ├──run_eval_gpu.sh          // GPU处理器评估的shell脚本
        │   ├──run_eval_cpu.sh          // CPU处理器评估的shell脚本
        ├── src
        │   ├──dataloader.py            // 读取目标数据集
        │   ├──model.py                 // DANN网络架构
        │   ├──train_cell.py            // DANN训练模块
        ├── train.py                    // 训练脚本
        ├── eval.py                     // 评估脚本
        ├── export.py                   // 将checkpoint文件导出到air/mindir
        ├── config.yaml                 // 配置文件
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置DANN。

  ```python
  'batch_size':64                                       # 训练批次大小
  'lr':0.001                                            # backbone预训练、分类层的学习率
  'lr_backbone_s':0.002                                 # backbone在对抗训练时的学习率
  'model_root':"./models"                               # 模型保存路径
  'n_epoch':100   # 对抗训练epoch数
  'n_pretrain_epochs':100                               # 预训练epoch数
  'source_dataset_name':"MNIST"                         # 源域数据集名称
  'source_image_root': "./dataset/MNIST"                # 源域数据集路径
  'target_dataset_name':"mnist_m"                       # 目标域数据集名称
  'target_image_root': "./dataset/mnist_m"              # 目标域数据集路径
  'weight_decay':1.0e-05                                # 权重衰减
  'device_target':"Ascend"                              # 运行设备
  'device_id':0                                         # 用于训练或评估数据集的设备ID使用run_train.sh进行分布式训练时可以忽略。
  'backbone_ckpt_file':"./models/best_backbone_t.ckpt"  # 推理时加载backbone的checkpoint文件的绝对路径
  'classifier_ckpt_file':"./models/best_class_classifier.ckpt"  # 推理时加载分类器的checkpoint文件的绝对路径
  'imageSize': 28                                       # 输入的图像大小
  'Linear': 800                                         # 全连接层输入的大小
  'file_name':"DANN.air"                                # 导出的文件名
  'file_format':"AIR"                                   # 导出的文件格式
  ```

更多配置细节请参考配置文件`config.yaml`。

## 训练过程  
  
### 训练  
  
- Ascend处理器环境运行  
  
  ```bash  
  python train.py > train.log 2>&1 &  
  ```  
  
  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式获得损失值：  
  ```bash  
  # grep "loss is " train.log  
  epoch: 0, [iter: 91 / all 921],  err_D_domain: 1.405273, err_G_domain: 0.480713,err_sum:1.885742  
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
  
  请将配置文件config.yaml中的`device_target`从`Ascend`改为`GPU`  
  
  ```bash  
  python eval.py > eval.log 2>&1 &  
  或  
  bash run_eval_gpu.sh  
  ```  
- 在CPU处理器环境运行时评估目标域数据集  
  
  请将配置文件config.yaml中的`device_target`从`Ascend`改为`CPU`  
  
  ```bash  
  python eval.py > eval.log 2>&1 &  
  或  
  bash run_eval_cpu.sh  
  ```  
    
  上述所有命令将在后台运行，您可以通过`eval/eval.log`文件查看结果。测试数据集的准确性如下：  
  
  ```bash  
  source domain accuracy: 0.994591
  target domain accuracy: 0.815737 
  ```  
  
## 导出过程  
  
### 导出  
  
  ```shell  
  python export.py
  ```  


# 模型描述

## 性能

### 训练性能

#### MNIST和MNIST_M上训练DANN

| 参数          | Ascend                                                                                         
|-------------|------------------------------------------------------------------------------------------------
| 模型版本        | Inception V1                                                                                   
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8                                                
| 上传日期        | 2023-07-04                                                                                     
| MindSpore版本 | 1.8.1                                                                                          
| 数据集         | MNIST,MNIST_M                                                                                  
| 训练参数        | batch_size: 64,lr: 0.001,lr_backbone_s: 0.002,n_epoch: 100,n_pretrain_epochs: 50               
| 优化器         | Adam                                                                                           
| 损失函数        | Softmax交叉熵                                                                                     
| 输出          | 概率                                                                                             
| 速度          | 单卡：130毫秒/步                                                                     
| 总时长         | 单卡：204 分钟                                                                     
| 参数(M)       | 0.176                                                                                           
| 微调检查点       | 0.696M (.ckpt文件)                                                                               
| 推理模型        | 0.714M (.mindir文件),  0.732M(.air文件)                                                              
| 脚本          | [googlenet脚本](https://gitee.com/mindspore/mindspore/tree/r0.7/model_zoo/official/cv/googlenet) 

### 评估性能

#### MNIST和MNIST_M上评估DANN

| 参数          | Ascend                 
| ------------------- |------------------------
| 模型版本       | Inception V1           
| 资源            | Ascend 910；系统 Euler2.8 
| 上传日期       | 2023-07-04             
| MindSpore 版本   | 1.8.1                  
| 数据集             | MNIST_M, 9000张图像       
| batch_size          | 64                     
| 输出             | 概率                     
| 准确性            | 单卡: 81.5% 
| 推理模型 | 0.714M (.mindir文件)       

# 随机情况说明

我们设置了train.py中的随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
