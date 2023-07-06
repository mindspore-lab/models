# 项目描述

本项目着力于复现 DNN 网络，通过使用mindspore框架对网络进行复现。

## DNN描述

DNN对于遥感图像配准问题是一种有效的深度神经网络。与分别进行特征提取和特征匹配的传统方法不同，
DNN网络从感知图像和参考图像中对补丁配对，然后直接学习这些补丁对与其匹配标签之间的映射，以便进行后期配准。
这种端到端架构允许网络在训练网络时通过在传统方法中缺乏的信息反馈优化整个处理（学习映射函数）。
此外，为了缓解遥感图像训练数据的小数据问题，DNN引入了一种自学习，利用图像及其转换后的副本学习映射函数。

## 模型结构描述

DNN网络模型

- 由全连接层构成。隐藏层的神经元个数分别为1568，784，100。输出层的神经元个数为1。
- 采用三层限制玻尔兹曼机（RBM）进行无监督的预训练来初始化DNN参数，层数分别为1568，784，100。
- 采用BP算法来有监督微调整个网络的参数。

## 数据集描述

包含六对图像，训练样本通过图像经过尺度和旋转变换以后裁块生成，其中尺度参数在[0.5, 1.5]范围内随机采样。

**Mexico City**：

由2000年4月和2002年5月的Landsat-7遥感卫星获取。它们的尺寸分别为512×512和502×502。

**Mediterranean Sardinia, Italy**：

由1995年9月和1996年7月的Landsat-5遥感卫星获取。它们的尺寸分别为412×300和404×294。

**Ottawa**：

为1997年5月至8月雷达卫星获取的渥太华市的SAR图像。它们的尺寸分别为290×350和284×343。它们由位于加拿大渥太华的加拿大国防研究与发展局提供。

**Mountain**：

由Thematic Mapper (TM)模拟器获得的山区上空的图像。它们是从一组12波段的航空图像中选择的，其中参考图像从波段1中剪切，经过旋转变换后从波段9中选择感测图像。测试图像的大小相同，为512×512。

**Mountain-river**：

Radarsat-2在2017年4月和2017年8月获得的图像。大小分别为500×512和444×512。由中国科学院遥感与地球研究所提供。

**Brasilia, Brazil**：

它们分别于1994年6月由遥感器陆地卫星专题制图器和1995年8月由SPOT拍摄。它们的大小相同，为256×256。值得注意的是，巴西的图像是由不同的传感器获取的。

**Yellow River Estuary**：

为Radarsat-2卫星在2008年6月和2009年6月获取的中国黄河口图像，大小相同，为1000×1000。

## 指标描述

控制点的数量(Nred)，基于所有控制点的均方根误差并归一化为像素大小(RMSall)，基于留一法(RMSLoo)控制点残差计算的均方根误差，象限残差分布的统计评估(Pquard)，范数高于 1.0(BPP(1:0)的坏点比例，关于残差散点图(Skew)上偏好轴的存在的统计评估，图像(Scat)中控制点分布的好坏的统计评估。这些指标值越小，性能越好，除了(Nred)。此外，棋盘拼接图像用于定性评估，其中图像边缘和区域重叠的连续性说明了配准性能。


## 性能描述

| 参数          | Ascend  | GPU     |
| ------------- |---------|---------|
| 模型版本      | DNN     | DNN     |
| 资源          |         |         |
| 上传日期      |         |         |
| MindSpore版本 | 1.7.0   | 1.7.0   |
| 数据集        |         |         |
| 训练参数      |         |         |
| 优化器        | SGD     | SGD     |
| 损失函数      | RMSE    | RMSE    |
| 速度          | 20毫秒/步; | 20毫秒/步; |
| 总时长        | xxx分钟;  | xxx分钟;  |
| 推理模型      | [模型]()  |         |

## 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 项目目录结构

```
.
|-- README.md
|-- config.yaml
|-- model_utils
|   |-- __init__.py
|   |-- config.py
|-- scripts
|   |-- run_train.sh
|-- src
|   |-- model
|   |   |-- var_init.py
|   |   |-- DBN.py
|   |   |-- nn.py
|   |   |-- RBM.py
|   |-- dataset.py
|   `-- Utils.py
|-- train.py
```

# 脚本说明

此部分对于代码训练、验证、测试等部分所使用的脚本命令进行讲解，并且展示相对应的效果。

## 训练脚本

- 所有的训练脚本都保存在train.sh文件当中，如果需要可以自行查阅，并且选取进行修改。

- 所有的模型及其训练日志文件可以通过[网盘链接]()直接进行下载。

- 其训练打印**日志案例**如下所示:

```
epoch:[  1/200], MSEloss: 0.124393, per step time:15.004 ms : : 836it [00:19, 43.90it/s]
epoch:[  2/200], MSEloss: 0.094920, per step time:18.003 ms : : 836it [00:18, 44.44it/s]
epoch:[  3/200], MSEloss: 0.087708, per step time:16.004 ms : : 836it [00:18, 45.05it/s]
epoch:[  4/200], MSEloss: 0.085392, per step time:15.003 ms : : 836it [00:21, 38.93it/s]
epoch:[  5/200], MSEloss: 0.084607, per step time:16.004 ms : : 836it [00:19, 43.46it/s]
...
```

### 主要调用的训练命令

```shell
python train.py --ckpt_save_dir ${checkpoint} \
                --data_dir ${DATA_PATH} \
                ----config_path ${config_path} \
                 >${checkpoint}"/device$i""/Train_device$i.log" &
```

**config_path中重点参数及其对应含义**

| 重点参数             | 含义             |
|------------------|----------------|
| ckpt_save_dir     | 模型保存位置         |
| DBN_numepochs    | DBN训练代数        |
| NN_numepochs     | NN网络训练代数       |
| batch-size       | 训练batch包含的图片数目 |
| lr     | 学习率            |
| momentum | 动量             |
| num_train     | 数据集中图片的数量      |
| device           | 训练所使用的设备       |

### 训练DNN网络模型

```shell
bash run_train.sh [DEVICE_NUM] [DATA_PATH]
```

## 推理脚本

### 测试权重文件

使用以下脚本，可以在工作目录下生成`result`文件夹,在result文件夹下可以生成对应的图片，可以观察输出结果的可视化效果。

并且脚本当中内置函数可以根据结果直接计算得出相应的RMSE指标。

```shell
bash run_eval.sh [MODEL_PATH] [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)]
```

### 计算输出RMSE

**运行脚本**:

```shell
python compute_iou.py data/cityscapes/gtFine/val result/cityscapes
```

**输出结果**:

```

```
