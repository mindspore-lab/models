# 目录

<!-- TOC -->

- [目录](#目录)
- [LUNAR 描述](#LUNAR描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)


# LUNAR描述

LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks方法于2022年提出，LUNAR 学会以可训练的方式利用每个节点最近邻居的信息来发现异常。比现有的局部离群值方法以及最先进的深基线表现得更好。对局部邻域大小的不同设置具有更强的鲁棒性。

[论文]([https://arxiv.org/abs/2112.05355])：Goodge A, Hooi B, Ng S K, et al. Lunar: Unifying local outlier detection methods via graph neural networks[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(6): 6737-6745.

# 模型架构

本文基于图神经网络（GNN）中使用的消息传递方案，在一个简单、通用的框架下统一局部异常值方法。我们证明了许多流行的方法，例如 KNN、LOF 和 DBSCAN，可以被视为这种更通用的消息传递框架的特例。尽管很受欢迎，但局部离群值方法缺乏学习优化或适应特定数据集的能力，例如通过可训练的参数。此外，在无监督的设置中，没有直接的方法来找到最佳的超参数设置，例如最近邻居的数量，这非常重要并且极大地影响性能。在本文中，我们还提出了一种名为 LUNAR（基于可学习统一邻域的异常排名）的新方法，该方法基于与局部异常值方法相同的消息传递框架，但通过图神经网络实现可学习性来解决其缺点。


# 数据集
公开数据集HRSS
获取方式(https://github.com/agoodge/LUNAR/tree/main)
下载地址中的data.zip，解压即可

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

# 脚本说明
## 脚本及样例代码

  ```text
├──LUNAR.py     #  GNN模型和训练程序
|——utils.py     #  用于加载数据集、预处理、图形构建、负采样的功能
|——variables.py  #超参数
|——main.py       #主文件

  ``` 

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python3 main.py --dataset HRSS --samples MIXED --k 100 --train_new_model
  ```

## 评估过程

### 评估  

- 在Ascend环境运行评估
  ```bash
  python3 main.py --dataset HRSS --samples MIXED --k 100
  ```

# 随机情况说明

main.py中设置了不同的随机数种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
