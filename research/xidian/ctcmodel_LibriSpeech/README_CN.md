# 目录

<!-- TOC -->

- [目录](#目录)
- [CTCModel介绍](#ctcmodel介绍)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
  - [脚本及样例代码](#脚本及样例代码)
  - [脚本参数](#脚本参数)
- [数据预处理过程](#数据预处理过程)
  - [数据预处理](#数据预处理)
- [训练过程](#训练过程)
  - [训练](#训练)
- [评估过程](#评估过程)
  - [评估](#评估)
- [模型描述](#模型描述)
  - [性能](#性能)
    - [训练性能](#训练性能)
    - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)

# CTCModel介绍

CTCModel利用CTC准则训练RNN模型，完成语素标记任务。CTC 的全称是Connectionist Temporal Classification，中文名称是“连接时序分类”，这个方法主要是解决神经网络label 和output 不对齐的问题，其优点是不用强制对齐标签且标签可变长，仅需输入序列和监督标签序列即可进行训练。
CTC被广泛的应用在语音识别，OCR等任务上，取得了显著的效果。

[论文](https://www.cs.toronto.edu/~graves/icml_2006.pdf): Alex Graves, Santiago Fernández, Faustino J. Gomez, Jürgen Schmidhuber:
Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. ICML 2006: 369-376

# 模型架构

模型包括  
一个两层的双向LSTM模型，输入维度为39，即提取出的语音特征的维度  
一个全连接层，输出维度为62,标签数+1,61代表空白符号

# 数据集

使用的数据集为: [Mini LibriSpeech](<https://www.openslr.org/31/>)，包含FLAC,TXT两种格式的文件
对下载解压后的数据预处理:

- 读取原始数据，将语音数据转化为wav格式
- 读取原始数据，将文本数据拆分为但数据的txt标签
- 读取语音数据和标签数据，通过mfcc和二阶差分提取语音信号特征
- 对处理后的数据进行填充，并将处理后的数据转化为MindRecord格式
- 这里提供了数据预处理的脚本preprocess_data.sh，将在后面数据预处理脚本部分详细介绍
- 预处理后的训练集长度为1519,测试集长度为1089

# 环境要求

- 硬件（ASCEND）
    - ASCEND处理器
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 通过下面网址可以获得更多信息:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html)
- 依赖
    - 见requirements.txt文件，使用方法如下:

```python
pip install -r requirements.txt
 ```

# 脚本说明

## 脚本及样例代码

```path
CTCModel
├── scripts
│   ├── eval.sh                        #评估脚本
│   ├── preprocess_data.sh             #预处理数据脚本
│   └── train_alone.sh                 #单卡训练脚本
├── src
│   ├── dataset.py                     #加载数据
│   ├── eval_callback.py               #边训练边测试
│   ├── loss.py                        #自定义损失函数
│   ├── metric.py                      #自定义指标
│   ├── model_for_eval.py              #自定义评价网络
│   ├── model_for_train.py             #自定义训练网络
│   ├── model.py                       #模型骨干文件
│   └── model_utils
│       ├── config.py                 #解析配置文件
│       ├── device_adapter.py         #区分本地/modelarts文件
│       ├── __init__.py
│       ├── local_adapter.py           #本地训练获取设备信息
│       └── moxing_adapter.py          #model arts配置，交换文件
├── default_config.yaml                 #参数配置文件
├── eval.py                             #评估网络
├── export.py                           #导出MINDIR格式
├── preprocess_Libri.py                 #将Libri_min数据处理
├── preprocess_data.py                  #预处理数据
└── train.py                            #训练网络
```

## 脚本参数

数据预处理、训练、评估的相关参数在`default_config.yaml`文件

```text
数据预处理相关参数
dataset_dir      保存预处理得到的MindRecord文件的目录
train_dir        预处理前的原始训练数据的目录
test_dir         预处理前的原始测试数据的目录
train_name       预处理后的训练MindRecord文件的名称
test_name        预处理后的测试MindRecord文件的名称  
```

```text
模型相关参数
feature_dim               输入特征维度，与预处理后的数据维度一致,39
batch_size                batch大小
hidden_size               隐藏层维度
n_class                   标签数，模型最后输出的维度,62
n_layer                   LSTM层数
max_sequence_length       序列最大长度，所有序列都填充到这一长度,1555
max_label_length          标签最大长度，所有标签都填充到这一长度,75
```

```text
训练相关参数
train_path                 训练集MindReord文件
test_path                  测试集MindRecord文件
save_dir                   保存模型的目录
epoch                      迭代轮次
lr_init                    初始学习率
clip_value                 梯度裁剪阈值
save_check                 是否保存模型
save_checkpoint_steps      保存模型的步数
keep_checkpoint_max        最大保存模型的数量
train_eval                 是否边训练边测试
interval                   每隔多少步做一次测试
run_distribute             是否分布式训练
dataset_sink_mode          是否开启数据下沉
```

```text
评估相关参数
test_path                  测试集MindRecord文件
checkpoint_path            模型保存路径
test_batch_size            测试集batch大小
beam                       greedy decode(False)还是prefix beam decode(True)，默认为greedy decode
```

```text
export相关参数
file_name                   导出文件名
file_format                 导出文件格式，MINDIR
```

```text
配置相关参数
enable_modelarts             云上训练
device_traget                硬件，只支持ASCEND
device_id                    设备号
```

# 数据预处理过程

## 数据预处理

数据预处理之前请先确认安装python-speech-features库
运行示例:
首先运行preprocess_Libri.py文件，将Libri_min数据预处理

```text
python preprocess_data.py \
--dataset_dir ./dataset \
--train_dir /data/Libri_mini/TRAIN \
--test_dir /data/Libri_mini/TEST \
--train_name train.mindrecord \
--test_name test.mindrecord
参数:
    --dataset_dir        存储处理后的MindRecord文件的路径，默认为./dataset,会自动新建
    --train_dir          原始训练集数据所在目录
    --test_dir           原始测试集数据所在目录
    --train_name         生成的训练文件名称，默认为train.mindrecord
    --test_name          生成的测试文件名称，默认为test.mindrecord
    其他参数可以通过default_config.yaml文件设置
```
./data/TIMIT/data/lisa/data/timit/raw/Libri_mini/TEST
或者可以运行脚本:

```bash
bash scripts/preprocess_data.sh [DATASET_DIR] [TRAIN_DIR] [TEST_DIR]
```

三个参数均为必须,分别对应上面的 ```--dataset_dir,--train_dir,--test_dir```
数据预处理过程较慢

# 训练过程

## 训练

- ### 单卡训练

运行示例:

```text
python train.py \
--train_path ./dataset/train.mindrecord0 \
--test_path ./dataset/test.mindrecord0 \
--save_dir ./save \
--epoch 120 \
--train_eval True \
--interval 5 \
--device_id 0 > train.log 2>&1 &
参数:
    --train_path         训练集文件路径
    --test_path          测试集文件路径
    --save_dir           模型保存路径
    --epoch              迭代轮数
    --train_eval         是否边训练边测试
    --interval           测试间隔
    --device_id          设备号
    其他参数可以通过default_config.yaml文件设置
```

或者可以运行脚本:

```bash
bash scripts/train_alone.sh [TRAIN_PATH] [TEST_PATH] [SAVE_DIR] [DEVICE_ID]
```

四个参数均为必须,分别对应上面的 ```--train_path,--test_path,--save_dir,--device_id```
上述命令将在后台运行，可以通过train.log查看结果  
第一个epoch算子编译时间较长，约60分钟，之后每个epoch约7分钟

# 评估过程

## 评估

评估之前请确认安装edit-distance库
运行示例:

```text
python eval.py \
--test_path ./dataset/test.mindrecord0 \
--checkpoint_path ./save/best.ckpt \
--beam False \
--device_id 0 > eval.log 2>&1 &
参数:
    --test_path          测试集文件路径
    --checkpoint_path    加载模型的路径
    --device_id          设备号
    --beam               greedy解码还是prefix beam解码
    其他参数可以通过default_config.yaml文件设置
```

或者可以运行脚本:

```bash
bash scripts/eval.sh [TEST_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
```

3个参数均为必须,分别对应上面的 ```--test_path,--checkpoint_path,--device_id```
上述命令在后台运行，可以通过eval.log查看结果,测试结果如下

```text
greedy decode运行结果
{'ler': 0.3038}
```


# 模型描述

## 性能

### 训练性能

| 参数                 | CTCModel                                                      |
| -------------------------- | ---------------------------------------------------------------|
| 资源                   | Ascend910             |
| 上传日期              | 待定                                    |
| MindSpore版本           | 1.8.1                                                          |
| 数据集                    | Mini LibriSpeech，训练集长度1519                                                 |
| 训练参数       | epoch=300, batch_size = 64, lr_init=0.01,clip_value=5.0   |
| 优化器                  | Adam                                                           |
| 损失函数              | CTCLoss                                |
| 输出                    | LER                                                   |
| 损失值                       | 717.1                                                       |
| 运行速度                      | 7229.011 ms/step                                   |
| 训练总时间       | 约一天;                                                                             |

### 推理性能

| 参数                 | CTCModel                                                      |
| -------------------------- | ----------------------------------------------------------------|
| 资源                   | Ascend910                   |
| 上传日期              | 待定                                 |
| MindSpore版本          | 1.8.1                                                           |
| 数据集                    | Mini LibriSpeech，测试集大小1089                         |
| batch_size                 | 1                                                               |
| 输出                    | LER:0.9                       |

# 随机情况说明

随机性主要来自下面两点:

- 参数初始化
- 轮换数据集