# 目录

<!-- TOC -->
* [目录](#目录)
* [MatchNet描述](#matchnet描述)
  * [概述](#概述)
  * [论文](#论文)
* [数据集](#数据集)
* [环境要求](#环境要求)
* [快速入门](#快速入门)
* [脚本说明](#脚本说明)
  * [脚本及样例代码](#脚本及样例代码)
  * [训练过程](#训练过程)
    * [用法](#用法)
      * [Ascend处理器环境运行](#ascend处理器环境运行)
      * [训练时推理](#训练时推理)
    * [结果](#结果)
  * [评估过程](#评估过程)
    * [用法](#用法-1)
      * [Ascend处理器环境运行](#ascend处理器环境运行-1)
    * [结果](#结果-1)
  * [导出过程](#导出过程)
    * [导出](#导出)
* [模型描述](#模型描述)
  * [性能](#性能)
    * [评估性能](#评估性能)
      * [Brown上MatchNet的性能](#brown上matchnet的性能)
* [随机情况说明](#随机情况说明)
* [ModelZoo主页](#modelzoo主页)
<!-- TOC -->

# MatchNet描述

## 概述

MatchNet是一个基于patch的局部图像匹配网络。该网络通过一个共享权重的backbone提取输入pair的共有特征，并通过一个度量网络来判断两者是否匹配。

## 论文

[MatchNet: Unifying Feature and Metric Learning for Patch-Based Matching](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Han_MatchNet_Unifying_Feature_2015_CVPR_paper.pdf)

# 数据集

[Brown patch dataset](http://phototour.cs.washington.edu/patches/default.htm)

该数据集包含三个子集(notredame/liberty/yosemite)

- 数据集大小

    - liberty：450092个patch块
    - notredame：468159个patch块
    - yosemite：633587个patch块

    每个patch大小为64 x 64像素。

- 数据格式：bmp文件

-   下载数据集。

    -   运行

    ```
    sh scripts/download_dataset.sh
    ```

    

-   准备训练数据清单文件。清单文件用于保存图片和标注对的相对路径。如下：

    ```shell
    $DATASET
    ├── liberty                             # liberty数据集根目录
    ├── notredame                           # notredame数据集根目录
    └─── yosemite                           # yosemite数据集根目录
    ```

配置并运行`src/build_data.sh`，将数据集转换为MindRecords。scripts/build_data.sh中的参数：

```shell
--data_root                 训练数据的根路径
--dst_path                  MindRecord所在路径
```

例如：

```shell
sh script/get_mindrecord.sh --dataroot=./Dataset --dst_path=./Mindrecord
```

# 环境要求

-   硬件（Ascend）

    -   准备Ascend处理器搭建硬件环境。
-   框架

    -   [MindSpore](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Finstall)
-   如需查看详情，请参见如下资源：

    -   [MindSpore教程](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Ftutorials%2Fzh-CN%2Fmaster%2Findex.html)
    -   [MindSpore Python API](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Fdocs%2Fzh-CN%2Fmaster%2Findex.html)
-   安装requirements.txt中的python包。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```bash
# 分布式训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]

# 分布式训练，从指定epoch开始恢复训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]

# 单机训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]

# 单机训练，从指定epoch开始恢复训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]

# 运行评估
bash scripts/run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

如果要在ModelArts上进行模型的训练，可以参考ModelArts的 [官方指导文档](https://support.huaweicloud.com/modelarts/) 开始进行模型的训练和推理，具体操作如下：

```text
# 训练模型
1. 创建作业
2. 选择数据集存储位置
3. 选择输出存储位置
2. 在模型参数列表位置按如下形式添加参数：
    data_url            [自动填充]
    train_url           [自动填充]
    dataset             [notredame/liberty/yosemite]
    checkpoint_url      [CHECKPOINT_PATH_OBS]
    modelarts           True
    run_distribute      [True/False]
    begin_epoch         [起始周期]
    eval                [True/False]
    evalset             [推理所用数据集]
3. 选择相应数量的处理器
4. 开始运行

# 评估模型
1. 创建作业
2. 选择数据集存储位置
3. 选择输出存储位置
2. 在模型参数列表位置按如下形式添加参数：
    data_url            [自动填充]
    train_url           [自动填充]
    checkpoint_url      [CHECKPOINT_PATH_OBS]
    modelarts           True
    dataset             [notredame/liberty/yosemite]
3. 选择单个处理器
4. 开始运行
```

# 脚本说明

## 脚本及样例代码

```text
MatchNet
├── modelarts                            # ModelArts训练相关脚本
│   └── start.py                         # ModelArts训练启动脚本
├── scripts                              # Ascend执行脚本
│   ├── get_mindrecord.sh                # 自动生成MindRecord训练文件
│   ├── run_distribute_train.sh          # 启动Ascend910分布式训练（8卡）
│   └── run_standalone_train.sh          # 启动Ascend910单机训练（单卡）
├── src
│   ├── callback.py                      # 自定义回调函数
│   ├── customfunc.py                    # 自定义forward类
│   ├── dataset.py                       # 数据集生成器
│   ├── make_mindrecord.py               # MindRecord文件生成
│   ├── MatchNet.py                      # MatchNet网络定义
│   └── var_init.py                      # 网络参数初始化函数
├── eval.py                              # 910推理脚本
├── export.py                            # 模型转换脚本
├── README.md
└── train.py                             # 910训练脚本
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```bash
# 分布式训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK] [DATASET_NAME] [EVAL_DATASET_NAME]

# 分布式训练，从指定周期开始恢复训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]   [DATASET_NAME] [EVAL_DATASET_NAME]

# 单机训练
Using: bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK] [DATASET_NAME] [EVAL_DATASET_NAME]

# 单机训练，从指定周期开始恢复训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]  [DATASET_NAME] [EVAL_DATASET_NAME]
```

#### 训练时推理

如果需要训练时推理，在执行shell脚本时为 `EVAL_CALLBACK` 参数传入 `True` 即可，其默认值为 `False` 。

### 结果

使用Brown数据集训练MatchNet

```text
# 分布式训练结果（8p）
epoch: 1 step: 1954, loss is 0.1930980086326599
Train epoch time: 183394.732 ms, per step time: 93.856 ms
Test FPR95: 0.2225165562913907
epoch: 2 step: 1954, loss is 0.09641222655773163
Train epoch time: 112773.925 ms, per step time: 57.714 ms
Test FPR95: 0.1410596026490066
···
epoch: 25 step: 1954, loss is 0.00016583407705184072
Train epoch time: 108716.832 ms, per step time: 55.638 ms
Test FPR95: 0.06821192052980131

Training finished, consume:7253.9462 ms
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 运行评估
bash scripts/run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

### 结果

评估结果保存在示例路径中，文件夹名为“eval”。你可在此路径下的日志文件中找到如下结果：

```text
Number of samples: 100352, total time: 63.17 ms
============= 910 Inference =============
FPR@95: 0.057692
=========================================
```

## 导出过程

### 导出

```bash
python export.py --device_id [DEVICE_ID] --checkpoint_file [CHECKPOINT_PATH] --file_name [OUTPUT_FILENAME] --file_format [OUTPUT_FILE_FORMAT]
```

# 模型描述

## 性能

### 评估性能

#### Brown上MatchNet的性能

| 参数          | Ascend 910                                               |
| ------------- | -------------------------------------------------------- |
| 模型版本      | MatchNet                                                 |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期      | 2023-7-5                                                 |
| MindSpore版本 | 1.8.1                                                    |
| 数据集        | Brown                                                    |
| 训练参数      | epoch=25, steps per epoch=1954, batch_size = 256         |
| 优化器        | SGD                                                      |
| 损失函数      | SoftmaxCrossEntropyWithLogits                            |
| 输出          | FPR@95                                                   |
| 损失          | 0.000166                                                 |
| 速度          | 60.934 ms/step（单卡）                                   |
| 总时长        | 2小时                                                    |

# 随机情况说明

`train.py`中使用了随机种子。

# ModelZoo主页

 请浏览官网 [主页](https://gitee.com/mindspore/models) 。

