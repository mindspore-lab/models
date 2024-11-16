# 目录

- [目录](#目录)
    - [FCN-4 介绍](#fcn-4-介绍)
    - [模型架构](#模型架构)
    - [环境依赖](#环境依赖)
    - [快速入门](#快速入门)
        - [1. 下载并准备数据集](#下载并准备数据集)
        - [2. 设置参数 (src/model_utils/default_config.yaml)](#设置参数)
        - [3. 训练](#测试)
        - [4. 测试](#训练)
    - [脚本描述](#脚本描述)
        - [脚本参数](#脚本参数)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [训练/推理性能](#推理性能)
    - [ModelZoo 主页](#ModelZoo 主页)

## [FCN-4 介绍](#目录)

该仓库提供了训练 FCN-4 模型的脚本和方法，以实现最新的精度表现

[论文](https://arxiv.org/abs/1606.00298):  `"Keunwoo Choi, George Fazekas, and Mark Sandler, “Automatic tagging using deep convolutional neural networks,” in International Society of Music Information Retrieval Conference. ISMIR, 2016."

## [模型架构](#目录)

FCN-4 是一种卷积神经网络架构，其名称 FCN-4 来源于它包含 4 层结构。这些层包括卷积层（Convolutional layers）、最大池化层（Max Pooling layers）、激活层（Activation layers）和全连接层（Fully connected layers）


## [环境依赖](#目录)

- 硬件（Ascend)
    - 准备带有Ascend处理器的环境.
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)

## [快速入门](#目录)

通过官网安装 MindSpore 后，您可以按照以下步骤开始训练和评估

### 1. 下载数据集

1. 下载分类数据集(GTAZN)链接: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download)
2. 提取数据集，在代码目录下创建一个data目录，将解压后的Data目录下的genres_original文件夹放在代码目录下的data目录，即fcn-4_GTZAN/data/genres_original

### 2. 设置参数 (src/model_utils/default_config.yaml)

### 3. 训练

在获得数据集后，首先使用以下代码将音频片段转换为 MindRecord 数据集：

```shell
python src/pre_process_data.py --device_id 0
```

然后，您可以使用以下代码开始训练模型：

```shell
python train.py --device_id 0
```

### 4. 测试

然后你可以测试你的模型

```shell
python eval.py --device_id 0
```
## [脚本描述](#目录)

### [脚本参数](#目录)

训练和评估的参数都可以在default_config.yaml中设置

- FCN-4的配置

  ```python

  'num_classes': 10                      # number of music genre classes
  'num_consumer': 4                      # file number for mindrecord
  'get_npy': 1 # mode for converting to npy, default 1 in this case
  'get_mindrecord': 1 # mode for converting npy file into mindrecord file，default 1 in this case
  'audio_path': "data/genres_original/" # path to audio clips
  'npy_path': "data/npy/" # path to numpy
  'info_path': "data/" # path to info_name, which provide the label of each audio clips
  'info_name': 'annotations_final.csv'   # info_name
  'device_target': 'Ascend'              # device running the program
  'device_id': 0                         # device ID used to train or evaluate the dataset. Ignore it when you use run_train.sh for distributed training
  'mr_path': "data/" # path to mindrecord
  'mr_name': ['train', 'val']            # mindrecord name

  'pre_trained': False                   # whether training based on the pre-trained model
  'lr': 0.0005                           # learning rate
  'batch_size': 32                       # training batch size
  'epoch_size': 10                       # total training epochs
  'loss_scale': 1024.0                   # loss scale
  'num_consumer': 4                      # file number for mindrecord
  'mixed_precision': False               # if use mix precision calculation
  'train_filename': 'train.mindrecord0'  # file name of the train mindrecord data
  'val_filename': 'val.mindrecord0'      # file name of the evaluation mindrecord data
  'data_dir': '/dev/data/Music_Tagger_Data/fea/' # directory of mindrecord data
  'device_target': 'Ascend'              # device running the program
  'device_id': 0,                        # device ID used to train or evaluate the dataset. Ignore it when you use run_train.sh for distributed training
  'keep_checkpoint_max': 10,             # only keep the last keep_checkpoint_max checkpoint
  'save_step': 2000,                     # steps for saving checkpoint
  'checkpoint_path': "weight/",  # the absolute full path to save the checkpoint file
  'prefix': 'MusicTagger',               # prefix of checkpoint
  'model_name': 'MusicTagger-10_24.ckpt', # checkpoint name
  ```

## [模型描述](#目录)

### [性能](#目录)

#### 训练/推理性能

| 参数                         | Ascend                                                                                           | 
|----------------------------|--------------------------------------------------------------------------------------------------| 
| 模型                         | FCN-4                                                                                            | 
| 资源                         | Ascend 910; CPU 2.60GHz, 56cores; Memory 314G; OS Euler2.8                                       | 
| 上传日期                       | 11/16/2024 (月/日/年)                                                                               | 
| MindSpore版本                | 2.2.14                                                                                           | 
| 数据集                        | GTZAN                                                                                            | 
| 训练参数                       | epoch=10, steps=534, batch_size = 32, lr=0.005                                                   |
| 优化器                        | Adam                                                                                             | 
| 损失函数                       | Binary cross entropy                                                                             | 
| 输出                         | probability                                                                                      | 
| 损失值                        | AUC 0.943                                                                                        |
| 运行速度                       | 1pc: 160 samples/sec;                                                                            | 
| 训练总时间                      | 1pc: 5 mins;                                                                                     |
| Checkpoint for Fine tuning | 198.73M(.ckpt file)                                                                              | 
| Scripts                    | [music_auto_tagging script](https://gitee.com/mindspore/models/tree/master/research/audio/fcn-4) |

## [ModelZoo 主页](#目录)  

官方 [主页](https://gitee.com/mindspore/models).  
