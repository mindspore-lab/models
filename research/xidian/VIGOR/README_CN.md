# 目录

<!-- TOC -->

* [目录](#目录)
* [VIGOR描述](#vigor描述)
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
    * [评估过程](#评估过程)
        * [用法](#用法-1)
            * [Ascend处理器环境运行](#ascend处理器环境运行-1)
    * [导出过程](#导出过程)
        * [导出](#导出)
* [模型描述](#模型描述)
* [随机情况说明](#随机情况说明)
* [ModelZoo主页](#modelzoo主页)
    <!-- TOC -->

# VIGOR描述

## 概述

VIGOR（VIGOR: Cross-View Image Geo-localization beyond One-to-one Retrieval）为经典的跨视角图像检索定位任务数据集，同时稳重基于所提出的VIGOR数据集进行训练和评估。本文中除了使用了基于Triplet检索损失对模型优化进行约束外，还引入了IoU损失以及回归损失，通过数据集中所提供的半正（Semi-positive）样本增强模型鲁棒性。


## 论文

[VIGOR: Cross-View Image Geo-localization beyond One-to-one Retrieval](https://ieeexplore.ieee.org/document/9578740/)

# 数据集

[VIGOR](https://github.com/Jeff-Zilence/VIGOR)

VIGOR收集了 90,618 张卫星参考图像和 105,214 张街景查询图像。此基准中的四个城市纽约（NewYork）、西雅图（Seattle）、旧金山（SanFrancisco）和芝加哥（Chicago）用于两种不同的分割设置：相同区域和跨区域。在相同设置中，所有城市的图像都用于训练和验证；在跨设置中，对纽约和西雅图进行训练，对旧金山和芝加哥进行评估，以测试该方法的泛化能力。另一个新颖之处是引入了所谓的半正图像。对于每一对正图像，都有三个半正卫星视图邻居，它们也覆盖街景图像的区域。拍摄它们的位置不在图像的中心区域。由于每张卫星图像都有这三张半正图像，因此很难获得较高的 R@1 分数。

- 数据集大小
    -   Chicago: 25, 479 张街景图像, 22, 308张卫星图像
    -   NewYork: 27, 769 张街景图像, 23, 279张卫星图像
    -   Seattle: 23, 751 张街景图像, 20, 776张卫星图像
    -   SanFrancisco: 28, 215 张街景图像, 24, 255张卫星图像

- 数据格式：jpg文件

-   下载数据集。

    请根据指引下载[VIGOR](https://github.com/Jeff-Zilence/VIGOR)。
    

    

-   下载完成后，请将数据集按照如下方式组织：

    ```shell
    $DATASET
    ├── Chicago
        ├── panorama
        └── satelite
    ├── NewYork
        ├─── ...
        └─── ...
    ├── Seattle
    ├── SanFrancisco
    └─── splits
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
    checkpoint_url      [CHECKPOINT_PATH_OBS]
    modelarts           True
    run_distribute      [True/False]
    begin_epoch         [起始周期]
    eval                [True/False]
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
3. 选择单个处理器
4. 开始运行
```

# 脚本说明

## 脚本及样例代码

```text
VIGOR
├── modelarts                            # ModelArts训练相关脚本
│   └── start.py                         # ModelArts训练启动脚本
├── scripts                              # Ascend执行脚本
训练文件
|   ├── run_eval.sh                      # 评估脚本
│   ├── run_distribute_train.sh          # 启动Ascend910分布式训练（8卡）
│   └── run_standalone_train.sh          # 启动Ascend910单机训练（单卡）
├── src
│   ├── callback.py                      # 自定义回调函数
│   ├── cal_delta.py                     # distance score计算
│   ├── dataset                          # 数据集加载
│   ├── loss                             # 自定义损失函数
│   ├── model                            # 模型结构
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
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK] 

# 分布式训练，从指定周期开始恢复训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]

# 单机训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]

# 单机训练，从指定周期开始恢复训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]
```

#### 训练时推理

如果需要训练时推理，在执行shell脚本时为 `EVAL_CALLBACK` 参数传入 `True` 即可，其默认值为 `False` 。


## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 运行评估
bash scripts/run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```


## 导出过程

### 导出

```bash
python export.py --device_id [DEVICE_ID] --checkpoint_file [CHECKPOINT_PATH] --file_name [OUTPUT_FILENAME] --file_format [OUTPUT_FILE_FORMAT]
```

# 模型描述

## 性能

### 评估性能

#### VIGOR dataset上VIGOR的性能

| 参数          | Ascend 910                                               |
| ------------- | -------------------------------------------------------- |
| 模型版本      | VIGOR                                                      |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期      | 2024-8-27                                                 |
| MindSpore版本 | 2.2.14                                                    |
| 数据集        | VIGOR                                              |
| 训练参数      | epoch=500, batch_size=32           |
| 优化器        | Adam                                                      |
| 损失函数      | SoftMarginTripletLoss, IoULoss, OffsetLoss |
| 输出          | Recall                                                 |                                                   |
| 速度          | 538.216 ms/step（单卡）                                  |

# 随机情况说明

`train.py`中使用了随机种子。

# ModelZoo主页

 请浏览官网 [主页](https://gitee.com/mindspore/models) 。