
# 目录

- [MKG Analogy概述](#MKG_Analogy概述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练过程](#训练过程)
    - [训练](#训练)
- [ModelZoo主页](#ModelZoo主页)

# MKG_Analogy概述

论文：[Multimodal Analogical Reasoning over Knowledge Graphs](https://arxiv.org/pdf/2210.00312.pdf)

<div align=center>
<img src="resource/mkg_analogy.gif" width="45%" height="45%" />
</div>

基于知识图谱的多模态类比推理致力于评估和探讨模型的在多模态场景下的类比推理能力，即给定一个类比问题，模型需要从知识图谱中找到类比问题的答案。类比问题的输入包括一个问题实体和一个问题模式，输出是一个答案实体。根据提供的模态信息的不同，多模态类比推理任务可以划分为单源类比推理和混源类比推理：

![img](resource/task.png)


# 数据集

To support the multimodal analogical reasoning task, we collect a multimodal knowledge graph dataset MarKG and a Multimodal Analogical ReaSoning dataset MARS. A visual outline of the data collection as shown in following figure:

为了支撑多模态类比推理任务，我们构建了一个多模态知识图谱数据集MarKG和一个多模态类比推理数据集MARS。数据收集的流程如下图所示：
![img](resource/flowchart.png)

在构建数据集时，我们遵循以下步骤：
1. 收集类比实体和关系
2. 链接到Wikidata并获取实体的结构信息
3. 获取和过滤图片信息
4. 采样类比推理数据

文本数据和结构信息放置在`dataset/`文件夹下, 图片数据可以通过 [Google Drive](https://drive.google.com/file/d/1AqnyrA05vKngfEbhw1mxY5qEoaqiKsC1/view?usp=share_link) 或[百度云盘(code:7hoc)](https://pan.baidu.com/s/1WZvpnTe8m0m-976xRrH90g)进行下载，并将下载后的图片放置在`dataset/MARS/images`路径下。

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend/GPU/CPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindsWe provide a knowledge graph to support and further divide the task into single and blended patterns. Note that the relation marked by dashed arrows (
) and the text around parentheses under images are only for annotation and not provided in the input.pore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html)

# 快速入门

通过官方网站安装MindSpore并下载数据集，之后在[huggingface](https://huggingface.co/)上下载CLIP模型的预训练权重和配置信息，并将下载后的权重转换为MindSpore模型，转换脚本为：


```shell
python convert_clip.py
```

# 脚本说明

## 脚本及样例代码

```text
mkg_analogy
    │── README.md   # 相关说明
    │── dataset
    │   │──MarKG        # 多模态知识图谱数据
    │   │──MARS         # 多模态类比推理数据
    │── scripts  # 运行脚本
    │── src
    │   │──data     # 数据处理代码
    │   │──models   # 模型代码
    │   │──utils    # 评估及其他相关代码
    │── convert_clip.py   # CLIP权重转换代码
    │── ms_main.py        # 主代码
```

## 脚本参数

在`scripts`文件夹下配置运行脚本及相关参数。

# 训练过程

## 训练

- GPU处理器环境或Ascend处理器运行

1. 基于MarKG预训练模型
    ```bash
    bash scripts/run_pretrain_mkgformer.sh
    ```

2. 基于MARS数据集微调模型

    将脚本中`pretrain`改为预训练后的模型权重路径，之后运行
    ```bash
    bash scripts/run_finetune_mkgformer.sh
    ```

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。

# 引用

```text
@inproceedings{
    zhang2023multimodal,
    title={Multimodal Analogical Reasoning over Knowledge Graphs},
    author={Ningyu Zhang and Lei Li and Xiang Chen and Xiaozhuan Liang and Shumin Deng and Huajun Chen},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=NRHajbzg8y0P}
}
```