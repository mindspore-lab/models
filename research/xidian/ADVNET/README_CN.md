# 项目描述

本项目着力于复现AdvNet网络，通过使用mindspore框架对网络进行复现。

## AdvNet描述

在AdvNet网络之前的一些域适应的工作是在feature层次上进行，在分割任务中效果不是非常好，这主要是因为分割任务中的高维的特征空间庞大且复杂，这导致了适应过程难以有效的进行，而且由于其复杂性，也导致了训练的时间和占用空间十分庞大。

文章通过观察数据的特点，观察到尽管源域和目标域在视觉外观上可能具有较大的差异，但是两者之间的分割预测图在空间高层语义类别分布上具备着一致性。因而在网络的输出空间当中进行域适应的方法是可行的。

## 模型结构描述

网络模型由主要两个模块构成：（1）分割网络G；（2）鉴别器网络D

1. 分割网络G

- 采用了基于ImageNet预训练的ResNet-101模型座作为基线网络，由于内存问题没有使用多尺度融合策略。
- 和其他工作类似，删除了最后分类层和修改最后两个卷积的步长从2到1,使输出特征图谱的分辨率为输入图像大小的1/8。
- 在conv4和conv5层使用了空洞卷积，补偿分别为2和4。
- 使用ASPP（Atrous Spatial Pyramid Pooling）作为最终的分类器
- 使用上采样层对softmax概率输出进行上采样，保持输入和输出的大小一致。

2. 鉴别器网络D

- 包含5个卷积层，其卷积核大小为4 x 4 ，步长为2。输出通道为[ 64 , 128 , 256 , 512 ,1]。
- 除了最后一层，每一个卷积层之后包括一个参数为0.2的Leaky ReLU激活函数。
- 最后一个卷积层之后增加了上采样层来保持输入和输出的大小一致。

多层结构当中，两个ASPP分别对conv4和conv5层提取到的特征进行直接分类。其对应的鉴别器网络也同样扩充为两个，具体结构如下图1所示：

![figure 1](./figure/AdvNet网络结构图.png)

图1：AdvNet网络结构图

## 数据集描述

点击[链接]()可跳转至网盘下载数据集。

**GTA5**：

数据集包括了24966张尺寸为1914×1052的图像。这些图像来源于基于洛杉矶真实地貌开发的GTA5游戏场景，并且最终经过人工标注而获得。图像中的场景和Cityscapes数据集中的场景高度相似，并且在图像标注上和Cityscapes数据集共享19个相同的类别。在本次复现中，我们使用了所有的GTA5数据集图像进行模型训练。

**Cityscapes**：

Cityscapes数据集包含了来自50个不同城市中驾驶场景的5000张图像，其中2957张图像用作训练，500张用作验证，1525张用作测试，尺寸为1024×2048。该数据集提供fine和coarse两种评测标准，前者提供5000张精标注图像，后者额外提供了20000张粗标注图像。

**数据集目录树**：

data  
╠══Cityscapes  
║      ╚══data  
║              ╠══gtFine  
║              ║     ╠══test  
║              ║     ╠══train  
║              ║     ╚══val  
║              ╚══leftimg8bit  
║                      ╠══test  
║                      ╠══train  
║                      ╚══val  
╚══GTA5  
        ╠══images  
        ╚══labels  

## 指标描述

语义分割任务会给每一个像素赋予其对应的类别，最终所有的类别预测都对应图像中的若干个区域。对于单一类别而言，类别预测区域和正确区域的交并比即该类的IoU指标。将所有类别的IoU值进行平均化处理，就可以得到最终图像预测的mIoU指标，该指标表示了模型最终的语义分割精度情况，其指标数值越高表明语义分割所达到的精度越高。

数学形式表达如式（1）所示，$i$其中表示对应的类别，$predict$表示对应类别的预测区域，$target$表示对应类别的真实区域：
$$
\begin{aligned}
IoU_i &= \frac{predict_i \cap target_i}{predict_i \cup target_i} \\
mIoU  &= \frac{1}{N}\sum_{i=1}^N IoU_i
\end{aligned}
$$

## 性能描述

| 参数          | Ascend        | GPU           |
| ------------- | ------------- | ------------- |
| 模型版本      | AdvNet        | AdvNet        |
| 资源          |               |               |
| 上传日期      |               |               |
| MindSpore版本 | 1.7.0         | 1.7.0         |
| 数据集        |               |               |
| 训练参数      |               |               |
| 优化器        | Momentum      | Momentum      |
| 损失函数      | Softmax交叉熵 | Softmax交叉熵 |
| 速度          | xxx毫秒/步;   | xxx毫秒/步;   |
| 总时长        | xxx分钟;      | xxx分钟;      |
| 推理模型      | [模型]()      |               |

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
|-- README_CN.md
|-- advnet_config.yaml
|-- ascend310_infer
|-- eval_adv.py
|-- export.py
|-- infer
|-- model_utils
|   |-- __init__.py
|   |-- __pycache__
|   |-- adaptive_pooling.py
|   |-- compute_iou.py
|   |-- config.py
|   |-- device_adapter.py
|   |-- local_adapter.py
|   |-- loss.py
|   |-- moxing_adapter.py
|   |-- pretrain
|   `-- softmax.py
|-- modelarts
|   |-- __init__.py
|   `-- start.py
|-- scripts
|   |-- compute.sh
|   |-- run_eval.sh
|   |-- run_train.sh
|   |-- run_train_cpu.sh
|   |-- run_train_gpu.sh
|   |-- test.sh
|   |-- train.sh
|   `-- train_src.sh
|-- src
|   |-- __init__.py
|   |-- advnet
|   |-- dataset
|   `-- utils
|-- train_adv.py
`-- train_src.py
```



# 脚本说明

此部分对于代码训练、验证、测试等部分所使用的脚本命令进行讲解，并且展示相对应的效果。

## 训练脚本

- 所有的训练脚本都保存在train.sh文件当中，如果需要可以自行查阅，并且选取进行修改。

- 所有的模型及其训练日志文件可以通过[网盘链接]()直接进行下载。

- 其训练打印**日志案例**如下所示:

```
iter =    11925/  250000, loss_seg1 = 0.274274 loss_seg2 = 0.194496 loss_adv1 = 2.495143, loss_adv2 = 3.229331 loss_D1 = 0.461276 loss_D2 = 0.389694 time:0.933252
iter =    11926/  250000, loss_seg1 = 0.328299 loss_seg2 = 0.204867 loss_adv1 = 2.810567, loss_adv2 = 3.073460 loss_D1 = 0.516043 loss_D2 = 0.584734 time:0.967968
iter =    11927/  250000, loss_seg1 = 0.452975 loss_seg2 = 0.270509 loss_adv1 = 3.714798, loss_adv2 = 4.076809 loss_D1 = 0.156671 loss_D2 = 0.117550 time:0.925966
iter =    11928/  250000, loss_seg1 = 0.177603 loss_seg2 = 0.118049 loss_adv1 = 2.997757, loss_adv2 = 3.326000 loss_D1 = 0.406134 loss_D2 = 0.352969 time:0.963190
iter =    11929/  250000, loss_seg1 = 0.271632 loss_seg2 = 0.199110 loss_adv1 = 3.225427, loss_adv2 = 3.712899 loss_D1 = 0.210772 loss_D2 = 0.179863 time:0.968328
iter =    11930/  250000, loss_seg1 = 0.309471 loss_seg2 = 0.212174 loss_adv1 = 2.780967, loss_adv2 = 2.835939 loss_D1 = 0.421679 loss_D2 = 0.458666 time:0.947201
...
```

### 主要调用的训练命令

```shell
python train_adv.py --snapshot_dir ${checkpoint} \
                    --data_dir ${GTA5_PATH} \
                    --data_dir_target ${CITYSCAPES_PATH} \
                    --restore_from /home/ma-user/work/model/Pretrain_DeeplabMulti.ckpt \
                    ----config_path ${config_path} \
                     >${checkpoint}"/device$i""/Train_device$i.log" &
```

**config_path中重点参数及其对应含义**

| 重点参数             | 含义                     |
| -------------------- | ------------------------ |
| snapshot-dir       | 模型保存位置             |
| lambda-seg         | 第一层辅助分割损失的权重 |
| lambda-adv-target1 | 第一层辅助对抗损失的权重 |
| lambda-adv-target2 | 第二层辅助对抗损失的权重 |
|batch-size|训练batch包含的图片数目|
| data-dir             | 源域数据路径             |
| data-dir-target      | 目标域数据路径           |
| input-size           | 网络接收的源域图像大小   |
| input-size-target    | 网络接受的目标域图像大小 |
| restore-from         | 加载预训练模型的路径     |
| device               | 训练所使用的设备         |
| continue_train | 断点续训对应的模型路径 |

### 训练Source Only模型

```shell
bash run_train_src.sh [DEVICE_NUM] [DATA_PATH]
```

### 训练AdvNet网络模型

```shell
bash run_train.sh [DEVICE_NUM] [DATA_PATH]
```

### 断点续训

支持从指定文件下直接读取对应模型进行断点续训，训练脚本可以在模型所在目录下寻找log文件，并且续写上log文件。

**Source Only：**

```
```

**AdvNet：**

```
```

## 推理脚本

### 测试权重文件

使用以下脚本，可以在工作目录下生成`result`文件夹,在result文件夹下可以生成对应的图片，可以观察输出结果的可视化效果。

并且脚本当中内置函数可以根据结果直接计算得出相应的mIoU指标。

```shell
bash run_eval.sh [MODEL_PATH] [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)]
```

### 计算输出mIoU

**运行脚本**:

```shell
python compute_iou.py data/cityscapes/gtFine/val result/cityscapes
```

**输出结果**:

```
Num classes 19
===>road:       81.24
===>sidewalk:   26.42
===>building:   77.75
===>wall:       18.75
===>fence:      23.12
===>pole:       21.78
===>light:      32.48
===>sign:       21.29
===>vegetation: 82.39
===>terrain:    34.4
===>sky:        76.84
===>person:     56.18
===>rider:      27.15
===>car:        77.92
===>truck:      39.66
===>bus:        44.68
===>train:      1.89
===>motocycle:  26.43
===>bicycle:    23.74
===> mIoU: 41.79
```

