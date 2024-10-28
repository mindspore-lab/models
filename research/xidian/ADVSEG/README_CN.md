# 模型结构描述

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

# 数据集描述

**[GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/)**：

数据集包括了24966张尺寸为1914×1052的图像。这些图像来源于基于洛杉矶真实地貌开发的GTA5游戏场景，并且最终经过人工标注而获得。图像中的场景和Cityscapes数据集中的场景高度相似，并且在图像标注上和Cityscapes数据集共享19个相同的类别。在本次复现中，我们使用了所有的GTA5数据集图像进行模型训练。

**[Cityscapes](https://www.cityscapes-dataset.com/)**：

Cityscapes数据集包含了来自50个不同城市中驾驶场景的5000张图像，其中2957张图像用作训练，500张用作验证，1525张用作测试，尺寸为1024×2048。该数据集提供fine和coarse两种评测标准，前者提供5000张精标注图像，后者额外提供了20000张粗标注图像。

**数据集目录树**：

```text
data  
╠══Cityscapes  
║   ╚══data  
║       ╠══gtFine  
║       ║   ╠══test  
║       ║   ╠══train  
║       ║   ╚══val  
║       ╚══leftimg8bit  
║           ╠══test  
║           ╠══train  
║           ╚══val  
╚══GTA5  
    ╠══images  
    ╚══labels  
```

# 指标描述

语义分割任务会给每一个像素赋予其对应的类别，最终所有的类别预测都对应图像中的若干个区域。对于单一类别而言，类别预测区域和正确区域的交并比即该类的IoU指标。将所有类别的IoU值进行平均化处理，就可以得到最终图像预测的mIoU指标，该指标表示了模型最终的语义分割精度情况，其指标数值越高表明语义分割所达到的精度越高。

数学形式表达如式（1）所示，$i$其中表示对应的类别，$predict$表示对应类别的预测区域，$target$表示对应类别的真实区域：
$$
\begin{aligned}
IoU_i &= \frac{predict_i \cap target_i}{predict_i \cup target_i} \\
mIoU &= \frac{1}{N}\sum_{i=1}^N IoU_i
\end{aligned}
$$

# 性能描述

| 参数          | Ascend                                                         | 
|-------------|----------------------------------------------------------------|
| 模型版本        | ADVSEG                                                         | 
| 资源          |                                                                | 
| 上传日期        |                                                                |   
| MindSpore版本 | 2.2.14                                                         | 
| 数据集         |                                                                |  
| 训练参数        |                                                                |     
| 优化器         | Momentum                                                       | 
| 损失函数        | Softmax交叉熵                                                     |
| 推理模型        | [模型](https://pan.baidu.com/s/1Vyotm7ZlPwC-2LCbmn1ATA?pwd=qoq3) | 

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# Train

- 所有的训练脚本都保存在train.sh文件当中，如果需要可以自行查阅，并且选取进行修改。

- 所有的模型及其训练日志文件可以通过[网盘链接](https://pan.baidu.com/s/1Vyotm7ZlPwC-2LCbmn1ATA?pwd=qoq3)直接进行下载。

- 其训练打印**日志案例**如下所示:

```
[Info:2024-03-18 06:10:18 | #Epoch:1/1 step:760/125000 | #avg_loss_G=0.1944 avg_loss_D1=0.4864 avg_loss_D2=0.5371 | #per time:5.732s ]: 750it [15:18,  1.66it/s]
[Info:2024-03-18 06:10:18 | #Epoch:1/1 step:760/125000 | #avg_loss_G=0.1944 avg_loss_D1=0.4864 avg_loss_D2=0.5371 | #per time:5.732s ]: 760it [15:18,  1.66it/s]
[Info:2024-03-18 06:10:24 | #Epoch:1/1 step:770/125000 | #avg_loss_G=0.2038 avg_loss_D1=0.4681 avg_loss_D2=0.5413 | #per time:5.729s ]: 760it [15:24,  1.66it/s]
...
```

```shell
python train.py --config_path "./advnet_config.yaml" --device_id [optional]
```

# Validation

使用以下脚本，可以在工作目录下生成`result`文件夹,在result文件夹下可以生成对应的图片，可以观察输出结果的可视化效果。

并且脚本当中内置函数可以根据结果直接计算得出相应的mIoU指标。

```shell
python eval.py --config_path "./advnet_config.yaml" \
               --restore_from [ckpt_path] \
               --device_id [optional]
```

```text
Acc/mIoU : 83.99%/41.77	 IoUs :
 [(0, 'road', 0.8116429128714054), (1, 'sidewalk', 0.263861659013041), (2, 'building', 0.7775755539600839), (3, 'wall', 0.18717109723889114), (4, 'fence', 0.23125807356266137), (5, 'pole', 0.2176798304732214), (6, 'traffic light', 0.3239562461825074), (7, 'traffic sign', 0.2123217038422464), (8, 'vegetation', 0.8240412751540129), (9, 'terrain', 0.34279919112493745), (10, 'sky', 0.7692770058602393), (11, 'person', 0.5616856074989725), (12, 'rider', 0.272045245396291), (13, 'car', 0.7781120606449263), (14, 'truck', 0.3966232707269801), (15, 'bus', 0.4465055601646668), (16, 'motorcycle', 0.018860930805899644), (17, 'bicycle', 0.26436523194640027)]
The iou is 41.77282810538131
```



