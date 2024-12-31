# 目录

<!-- TOC -->

* [目录](#目录)
* [ssda-mme描述](#ssda-mme描述)
    * [概述](#概述)
    * [论文](#论文)
* [数据集](#数据集-DomainNet)
* [环境要求](#环境要求)
* [快速入门](#快速入门)
* [脚本说明](#脚本说明)
    * [脚本及样例代码](#脚本及样例代码)
    * [训练过程](#训练过程)
    * [评估过程](#评估过程)
* [随机情况说明](#随机情况说明)
* [ModelZoo主页](#modelzoo主页)
    <!-- TOC -->

# SSDA-MME描述

## 概述
SSDA-MME(Semi-supervised Domain Adaptation via Minimax Entropy)
方法于2019年提出，旨在解决半监督域适应图像分类问题。
本文中作者提出了一种新颖的极小极大熵（MME）方法，
该方法通过对抗性优化来训练一个自适应的小样本模型。
适应过程是交替进行：
一方面针对分类器最大化未标记目标数据的条件熵，
另一方面针对特征编码器最小化该条件熵。
如此实现了源域和目标域的特征对齐。

## 论文

[Semi-supervised Domain Adaptation via Minimax Entropy](https://arxiv.org/pdf/1904.06487)

# 数据集-DomainNet
DomainNet数据集是一个大规模的领域适应（Domain Adaptation）基准数据集，
旨在评估和推动跨领域视觉识别算法的研究。
它是目前最大的领域适应数据集之一，
涵盖了多个领域和类别，为研究者提供了丰富的实验资源。以下是DomainNet数据集的详细介绍：

DomainNet包含6个不同的领域，
每个领域代表一种独特的视觉风格,
本文在选择如下四个域展开实验, 分别为:
- Real：真实世界的照片。
- Clipart：矢量剪贴画图像。
- Painting：艺术绘画作品。
- Sketch：手绘素描图像。

数据集下载方法为：
```bash
bash scripts/downdata.sh
```
txt文件夹下载链接为[github](https://github.com/VisionLearningGroup/SSDA_MME/tree/master/data/txt/multi).

数据集会自动下载并解压在 data/multi目录下，组织形式为：
```text
data
├── multi                # DomainNet数据集路径
│   ├── real             # Real域数据
|   ├── clipart          # clipart域数据
│   ├── painting         # painting域数据
│   ├── sketch          # sketch域数据
│   ├── txt             # 存储训练及测试样本名称的txt文件夹
│   │   ├── multi       
```

# 环境要求

-   硬件（Ascend）

    -   准备Ascend处理器搭建硬件环境。
-   框架

    -   [MindSpore](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Finstall)
-   如需查看详情，请参见如下资源：

    -   [MindSpore教程](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Ftutorials%2Fzh-CN%2Fmaster%2Findex.html)
    -   [MindSpore Python API](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Fdocs%2Fzh-CN%2Fmaster%2Findex.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```bash
# 训练
bash scripts/run_train.sh [DEVICE_NUM] [SOURCE_DOMAIN] [TARGET_DOMAIN] [SAMPLE_NUM]

# 运行评估
bash scripts/run_eval.sh [DEVICE_NUM] [SOURCE_DOMAIN] [TARGET_DOMAIN][SAMPLE_NUM]
```


# 脚本说明

## 脚本及样例代码

```text
SSDA-MME
├── base                              
│   ├── __init__.py                
│   ├── base_dataloader.py        # 数据加载实现  
│   ├── base_net.py               # 基础网络结构实现 
│   ├── base_train.py             #动态图训练实现 
│   ├── base_train_graph.py       #静态图训练实现       
│   ├── config.py                 #读取配置文件代码实现 
├── utils                               
│   ├── args2json.py                
│   ├── logger.py                
│   ├── return_dataset.py             
├── scripts                        # Ascend执行脚本
│   ├── downdata.sh                # 数据集下载脚本
│   ├── run_train.sh               # 启动Ascend910训练
│   ├── run_eval.sh               # 启动Ascend910验证
├── src
│   ├── callback.py                      # 自定义回调函数
│   ├── customfunc.py                    # 自定义forward类
│   ├── dataset.py                       # 数据集生成器
│   ├── make_mindrecord.py               # MindRecord文件生成
│   ├── svhn2mnist.py                    # 网络定义
│   └── var_init.py                      # 网络参数初始化函数
├── eval.py                              # 推理脚本
├── main.py                              # 训练脚本
├── resume.py                            # 重新恢复训练脚本
├── README.md
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```bash
# 训练
bash scripts/run_train.sh [DEVICE_NUM] [SOURCE_DOMAIN] [TARGET_DOMAIN][SAMPLE_NUM]
```


## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 运行评估
bash scripts/run_eval.sh [DEVICE_NUM] [SOURCE_DOMAIN] [TARGET_DOMAIN][SAMPLE_NUM]
```

# 随机情况说明

`main.py`中使用了随机种子。

# ModelZoo主页

 请浏览官网 [主页](https://gitee.com/mindspore/models) 。