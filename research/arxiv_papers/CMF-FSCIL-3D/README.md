#### 项目介绍

该项目是利用跨模态学习对点云识别增强的实现，该模型在论文 [💬On the Cross-Modality Fantasy for 3D Point Clouds Few-Shot Class-Incremental Learning] 中提出。项目实现完全基于 mindspore。

#### 简介

我们通过对公开数据集Shapenet、Co3d进行伪类数据生成，结合CLIP预训练模型强大的VT感知和理解能力，通过跨模态对比学习加强到Pointnet等三维点云识别的模型网络上，最后通过NCM分类器实现小样本类增量学习和预测

#### 用法

1. 项目准备：
   1. 下载此项目并解压缩
   2. `pip install -r requirement.txt`安装所需环境
   3. 我们使用的版本是`python=3.9 mindspore=2.2.14`

3. 运行准备：
   1. 在根目录下直接运行`python main.py`
   2. 如果需要修改运行config，可以到src/config.py中修改
