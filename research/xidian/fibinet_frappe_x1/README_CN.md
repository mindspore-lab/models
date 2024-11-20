# 目录

- [目录](#目录)
- [FiBiNET概述](#fibinet概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
  - [训练过程](#训练过程)
    - [单机训练](#单机训练)
  - [评估过程](#评估过程)
  - [推理过程](#推理过程)
    - [导出MindIR](#导出mindir)
    - [模型表现](#模型表现)
  - [脚本说明](#脚本说明)
  - [脚本和样例代码](#脚本和样例代码)
  - [脚本参数](#脚本参数)
    - [训练脚本参数](#训练脚本参数)
    - [预处理脚本参数](#预处理脚本参数)
- [模型描述](#模型描述)
  - [性能](#性能)
    - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
  - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# FiBiNET概述

FiBiNET (Feature Importance and Bilinear feature Interaction NETwork) 新浪微博于2019年提出的一种基于深度学习的广告推荐算法。[FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)论文中描述了FiBiNET的实现原理。

# 模型架构

FiBiNET模型训练了宽线性模型和深度学习神经网络，并在Wide&Deep的基础上对神经网络部分增加了动态学习特征重要性的SENET模块 (Squeeze-and-Excitation Network) 与学习特征交叉的Bilinear-Interaction模块。

# 数据集

- [Frappe_x1](https://pan.baidu.com/s/1QDMh8-JJCwPTaK0TUYft9A?pwd=jw8p)

# 环境要求

- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)，如需查看详情，请参见如下资源：
        - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
        - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html)

# 快速入门

1. 克隆代码。

```bash
git clone https://github.com/mindspore-lab/models.git
cd models/research/xidian/fibinet_frappe_x1
```

2. 下载数据集。

  > 请参考[1](#数据集)获得下载链接。

```bash
mkdir -p data-frappe/data && cd data-frappe/data
wget DATA_LINK
```

3. 使用此脚本预处理数据。生成的MindRecord数据存放在data-frappe/mindrecord路径下。

```bash
python src/preprocess_data.py  --data_path=./data-frappe/ --dense_dim=1 --slot_dim=9 --threshold=100 --train_line_count=202028 --skip_id_convert=0
```

4. 开始训练。

数据集准备就绪后，即可训练和评估模型。

## 训练过程

### 单机训练

运行如下命令训练模型：

```bash
python train.py --data_path=./data-frappe/mindrecord --dataset_type=mindrecord --device_target=Ascend
```

## 评估过程

运行如下命令单独评估模型：

```bash
python eval.py --data_path=./data-frappe/mindrecord --dataset_type=mindrecord --device_target=Ascend --ckpt_path=./ckpt/fibinet_train-10_897.ckpt
```

## 推理过程

### [导出MindIR](#contents)

```bash
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --device_target [DEVICE_TARGET] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，默认值："./ckpt/fibinet_train-10_41265.ckpt"；

`FILE_FORMAT` 必须在 ["AIR", "MINDIR"]中选择，默认值："MINDIR"。

### 模型表现

推理结果保存在脚本执行的当前路径，在eval_output.log中可以看到以下精度计算结果。

```markdown
auc :  0.8864351622697823
```

## 脚本说明

## 脚本和样例代码

```markdown
└── fibinet
    ├── README.md                                 # 所有fibinet模型相关说明与教程
    ├── requirements.txt                          # python环境
    ├── script
    │   ├── common.sh
    │   ├── run_train_gpu.sh                      # GPU处理器单卡训练shell脚本
    │   └── run_eval_gpu.sh                       # GPU处理器单卡评估shell脚本
    ├──src
    │   ├── callbacks.py
    │   ├── datasets.py                           # 创建数据集
    │   ├── generate_synthetic_data.py            # 生成虚拟数据
    │   ├── __init__.py
    │   ├── metrics.py                            # 模型表现评价指标脚本
    │   ├── preprocess_data.py                    # 数据预处理
    │   ├── process_data.py
    │   ├── fibinet.py                            # FiBiNET主体架构
    │   └── model_utils
    │       ├── __init__.py
    │       ├── config.py                         # 获取训练配置信息
    │       └── moxing_adapter.py                 # 参数处理
    ├── default_config.yaml                       # 训练参数配置文件，任何模型相关的参数均建议在此处修改
    ├── train.py                                  # 训练脚本
    ├── eval.py                                   # 评估脚本
    └── export.py
```

## 脚本参数

### 训练脚本参数

```markdown

Used by: train.py

Arguments:

  --device_target                     Device where the code will be implemented, only support GPU currently. (Default:GPU)
  --data_path                         Where the preprocessed data is put in
  --epochs                            Total train epochs. (Default:10)
  --full_batch                        Enable loading the full batch. (Default:False)
  --batch_size                        Training batch size.(Default:1000)
  --eval_batch_size                   Eval batch size.(Default:1000)
  --line_per_sample                   The number of sample per line, must be divisible by batch_size.(Default:10)
  --field_size                        The number of features.(Default:39)
  --vocab_size                        The total features of dataset.(Default:200000)
  --emb_dim                           The dense embedding dimension of sparse feature.(Default:10)
  --deep_layer_dim                    The dimension of all deep layers.(Default:[400,400,400])
  --deep_layer_act                    The activation function of all deep layers.(Default:'relu')
  --keep_prob                         The keep rate in dropout layer.(Default:0.5)
  --dropout_flag                      Enable dropout.(Default:0)
  --output_path                       Deprecated
  --ckpt_path                         The location of the checkpoint file. If the checkpoint file
                                      is a slice of weight, multiple checkpoint files need to be
                                      transferred. Use ';' to separate them and sort them in sequence
                                      like "./checkpoints/0.ckpt;./checkpoints/1.ckpt".
                                      (Default:"./ckpt/")
  --eval_file_name                    Eval output file.(Default:eval.og)
  --loss_file_name                    Loss output file.(Default:loss.log)
  --dataset_type                      The data type of the training files, chosen from [tfrecord, mindrecord, hd5].(Default:mindrecord)
  --vocab_cache_size                  Enable cache mode.(Default:0)
  --eval_while_train                  Whether to evaluate after training each epoch
```

### 预处理脚本参数

```markdown

used by: generate_synthetic_data.py

Arguments:
  --output_file                        The output path of the generated file.(Default: ./train.txt)
  --label_dim                          The label category. (Default:2)
  --number_examples                    The row numbers of the generated file. (Default:4000000)
  --dense_dim                          The number of the continue feature.(Default:13)
  --slot_dim                           The number of the category features.(Default:26)
  --vocabulary_size                    The vocabulary size of the total dataset.(Default:400000000)
  --random_slot_values                 0 or 1. If 1, the id is generated by the random. If 0, the id is set by the row_index mod
                                       part_size, where part_size is the vocab size for each slot
```

```markdown

usage: preprocess_data.py

  --preprocess_data_path              Where the origin sample data is put in (i.e. where the file origin_data is put in)
  --dense_dim                         The number of your continues fields.(default: 13)
  --slot_dim                          The number of your sparse fields, it can also be called category features.(default: 26)
  --threshold                         Word frequency below this value will be regarded as OOV. It aims to reduce the vocab size.(default: 100)
  --train_line_count                  The number of examples in your dataset.
  --skip_id_convert                   0 or 1. If set 1, the code will skip the id convert, regarding the original id as the final id.(default: 0)
  --eval_size                         The percent of eval samples in the whole dataset.
  --line_per_sample                   The number of sample per line, must be divisible by batch_size.
```

# 模型描述

## 性能

### 评估性能

| 计算框架               | MindSpore                                      |
| ---------------------- |------------------------------------------------|
| 处理器                 | Ascend                                         |
| MindSpore版本        | 1.8                                            |
| 数据集                  | [1](#数据集)                                      |
| 训练参数      | Epoch=10,<br />batch_size=1000,<br />lr=0.0001 |
| 优化器                | FTRL,Adam                                      |
| 损失函数       | Sigmoid交叉熵                                     |
| AUC分数        | 0.8864351622697823                             |
| 速度           | 7290.011毫秒/步                                     |
| 损失           | 0.3053886                                      |

所有可执行脚本参见[此处](https://gitee.com/mindspore/models/tree/master/research/recommend/fibinet/script)。

# 随机情况说明

以下三种随机情况：

- 数据集的打乱。
- 模型权重的随机初始化。
- dropout算子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
