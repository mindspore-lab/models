# 目录

<!-- TOC -->

- [目录](#目录)
    - [TB-Net概述](#tbnet概述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [文件列表](#文件列表)
        - [脚本参数](#脚本参数)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [评估性能](#评估性能)
            - [推理和解释性能](#推理和解释性能)
    - [随机情况说明](#随机情况说明)

# [TB-Net概述](#目录)

TB-Net是一个基于知识图谱的可解释推荐系统。

论文：Shendi Wang, Haoyang Li, Caleb Chen Cao, Xiao-Hui Li, Ng Ngai Fai, Jianxin Liu, Xun Xue, Hu Song, Jinyu Li, Guangye Gu, Lei Chen. [Tower Bridge Net (TB-Net): Bidirectional Knowledge Graph Aware Embedding Propagation for Explainable Recommender Systems](https://ieeexplore.ieee.org/document/9835387)

# [模型架构](#目录)

TB-Net将用户和物品的交互信息以及商品的属性信息在知识图谱中构建子图，并利用双向传导的计算方法对图谱中的路径进行计算，最后得到可解释的推荐结果。

# [数据集](#目录)

本示例提供Kaggle上的Steam游戏平台公开数据集，包含 [用户与游戏的交互记录](https://www.kaggle.com/tamber/steam-video-games) 和 [游戏的属性信息](https://www.kaggle.com/nikdavis/steam-store-games?select=steam.csv) 。

请参考 [下载数据集](https://www.mindspore.cn/xai/docs/zh-CN/master/using_tbnet.html#下载数据集) 以了解如何取得用例数据集及其文件格式。

# [环境要求](#目录)

- 硬件
    - 支持GPU。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
    - [MindSpore XAI](https://www.mindspore.cn/xai/docs/zh-CN/master/index.html)
- 如需查看详情，请参见如下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [快速入门](#目录)

请参考 [使用TB-Net](https://www.mindspore.cn/xai/docs/zh-CN/master/using_tbnet.html) 以了解如何快速入门。

# [脚本说明](#目录)

## [脚本和样例代码](#目录)

```text
.
└─tbnet
  ├─README.md
  ├─README_CN.md
  ├─data
  │ └─steam
  │   ├─config.json                 # 超参和训练配置
  │   ├─src_infer.csv               # 推理用原始数据
  │   ├─src_test.csv                # 测试用原始数据
  │   └─src_train.csv               # 训练用原始数据
  ├─src
  │ ├─dataset.py                    # 数据集加载器
  │ ├─embedding.py                  # 实体嵌入模组
  │ ├─metrics.py                    # 模型度量
  │ ├─path_gen.py                   # 数据预处理器
  │ ├─recommend.py                  # 推理结果集成器
  │ └─tbnet.py                      # TB-Net网络架构
  ├─export.py                       # 导出MINDIR/AIR文件脚本
  ├─preprocess.py                   # 数据预处理脚本
  ├─eval.py                         # 评估网络脚本
  ├─infer.py                        # 推理和解释脚本
  └─train.py                        # 训练网络脚本
```

## [脚本参数](#目录)

- preprocess.py 参数

```text
--dataset <DATASET>        目前只支持 'steam' 数据集 (默认 'steam')
--same_relation            只产生 relation1 和 relation2 相同的关系路径
```

- train.py 参数

```text
--dataset <DATASET>        目前只支持 'steam' 数据集 (默认 'steam')
--train_csv <TRAIN_CSV>    在数据集文件夹中的训练CSV文件名 (默认 'train.csv')
--test_csv <TEST_CSV>      在数据集文件夹中的测试CSV文件名 (默认 'test.csv')
--epochs <EPOCHS>          要训练的epochs数 (默认 20)
--device_id <DEVICE_ID>    GPU或Ascend卡 id (默认 0)
--run_mode <MODE>          模型运行模式，可选： 'GRAPH' 或 'PYNATIVE' (默认 'GRAPH')
```

- eval.py 参数

```text
--checkpoint_id <CKPT_ID>    要使用的 checkpoint(.ckpt) 文件的id
--dataset <DATASET>          目前只支持 'steam' 数据集 (默认 'steam')
--csv <CSV>                  在数据集文件夹中的测试CSV文件名 (默认 'test.csv')
--device_id <DEVICE_ID>      GPU或Ascend卡 id (默认 0)
--run_mode <MODE>            模型运行模式，可选： 'GRAPH' 或 'PYNATIVE' (默认 'GRAPH')
```

- infer.py 参数

```text
--checkpoint_id <CKPT_ID>    要使用的 checkpoint(.ckpt) 文件的id
--dataset <DATASET>          目前只支持 'steam' 数据集 (默认 'steam')
--csv <CSV>                  在数据集文件夹中的推理CSV文件名 (默認 'infer.csv')
--items <ITEMS>              要推荐的商品數量 (默认 3)
--explanations <EXP>         每個推荐商品要顯示的理據數量 (默认 3)
--device_id <DEVICE_ID>      GPU或Ascend卡 id (默认 0)
--run_mode <MODE>            模型运行模式，可选： 'GRAPH' 或 'PYNATIVE' (默认 'GRAPH')
```

- export.py 参数

```text
--config_path <CFG_PATH>        配置文档 (config.json) 路径
--checkpoint_path <CKPT_PATH>   checkpoint (.ckpt) 文档路径
--file_name <FILENAME>          导出的档案名称
--file_format <FORMAT>          导出的档案格式 'MINDIR' 或 'AIR' (默认 'MINDIR')
--device_id <DEVICE_ID>         GPU或Ascend卡 id (默认 0)
--run_mode <MODE>               模型运行模式，可选： 'GRAPH' 或 'PYNATIVE' (默认 'GRAPH')
```

# [模型描述](#目录)

## [性能](#目录)

### [训练性能](#目录)

| 参数                  | GPU                                                |
| -------------------  | --------------------------------------------------- |
| 模型版本              | TB-Net                                              |
| 资源                  |Tesla V100-SXM2-32GB                                 |
| 上传日期              | 2021-08-01                                          |
| MindSpore版本         | 1.3.0                                               |
| 数据集                | steam                                               |
| 训练参数              | epoch=20, batch_size=1024, lr=0.001                 |
| 优化器                | Adam                                                |
| 损失函数              | Sigmoid交叉熵                                        |
| 输出                  | AUC=0.8596，准确率=0.7761                            |
| 损失                  | 0.57                                               |
| 速度                  | 单卡：90毫秒/步                                      |
| 总时长                | 单卡：297秒                                          |
| 微调检查点             | 104.66M (.ckpt 文件)                                |

### [评估性能](#目录)

| 参数                        | GPU                          |
| -------------------------- | ----------------------------- |
| 模型版本                    | TB-Net                        |
| 资源                        | Tesla V100-SXM2-32GB         |
| 上传日期                    | 2021-08-01                    |
| MindSpore版本               | 1.3.0                         |
| 数据集                      | steam                         |
| 批次大小                    | 1024                          |
| 输出                        | AUC=0.8252，准确率=0.7503      |
| 总时长                      | 单卡：5.7秒                    |

### [推理和解释性能](#目录)

| 参数                        | GPU                           |
| -------------------------- | ----------------------------- |
| 模型版本                    | TB-Net                        |
| 资源                        | Tesla V100-SXM2-32GB          |
| 上传日期                    | 2021-08-01                     |
| MindSpore版本               | 1.3.0                         |
| 数据集                      | steam                         |
| 输出                        | 推荐结果和解释结果              |
| 总时长                      | 单卡：3.66秒                   |

# [随机情况说明](#目录)

- `tbnet.py`和`embedding.py`中Embedding矩阵的随机初始化。


## 豆瓣音乐评价数据集
### checkpoint
下载预训练权重[This link](https://pan.baidu.com/s/1cuQKkBkQsrty3yHVz1US9Q?pwd=3vaj) 放置于checkpoints/douban下

### 数据集
下载数据集[This link](https://pan.baidu.com/s/1PsnBjwo4kn3lODtIYPCLtQ?pwd=r2xt) 放置于data下
路径如下：
```text
data
└─douban
    ├─*.csv
    └─*.json
```
注意：该数据集为处理后的数据集，若需要原始数据集，请看此链接[This link](https://pan.baidu.com/s/1rmzAhf7Swc_Dj2ifi6hKWw?pwd=2mpu)
     下载后根据所需的词条自行生成所需的csv文件和json文件

### train
```bash
python train.py --dataset douban  --device_id=0 --epochs=20 --run_mode GRAPH
```
### eval
```bash
python eval.py --dataset douban --device_id=0 

```
### results
AUC:0.7045986763787052 ACC:0.8195219123505976


## movieLens-mini数据集
### checkpoint
下载预训练权重[This link](https://pan.baidu.com/s/1ckHGyrrIEluV7frL2ckX1A?pwd=iurd) 放置于checkpoints/ml-1m下

### 数据集
下载数据集[This link](https://pan.baidu.com/s/1rnPKU9L3UeI5fdFt-8ORNg?pwd=q7i2) 放置于data下
路径如下：
```text
data
└─ml-1m
    ├─*.csv
    └─*.json
    └─*.dat
    └─*.py
    └─README
```
注意：该数据集为movieLens-1m数据集采样10%并经过处理得到的数据集，其中*.dat文件为ml-1m原始文件，*.py文件为预处理文件，若有自定义需求，请自行更改

### train

```bash
python train.py --dataset ml-1m --device_id=0 --epochs=20 --run_mode GRAPH
```
### eval
```bash
python eval.py --dataset ml-1m --device_id=0 

```
### results
AUC:0.7936807583086596 ACC:0.7869528233985474

## Frappe数据集
### checkpoint
下载预训练权重[This link](https://pan.baidu.com/s/1-YFGnwo-833n9oCg3C_DNQ?pwd=wig4) 放置于checkpoints/frappe下
### 数据集
下载数据集[This link](https://pan.baidu.com/s/1r2yl6VxzXnVhMTeW6wPf5w?pwd=976a) 放置于data下
路径如下：
```text
data
└─frappe
    ├─*.csv
    └─*.json
```
### train
```bash
python train.py --dataset frappe --device_id=0 --epochs=20 --run_mode GRAPH
```
### eval
```bash
python eval.py --dataset frappe --device_id=0 
```

### results
AUC:0.8956322966273367 ACC:0.8803083187899942




