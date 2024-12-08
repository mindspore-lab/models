# [Tune it the Right Way: Unsupervised Validation of Domain Adaptationvia Soft Neighborhood Density (ICCV 2021)](https://arxiv.org/pdf/2108.10860.pdf)

#### [[Paper]](https://arxiv.org/pdf/2108.10860.pdf)

# 项目介绍

在这项工作中，我们的目标是建立一个可以以无监督的方式调整无监督域自适应模型的超参数的标准。

## 数据集描述

- 下载[GTA5数据集](https://download.visinf.tu-darmstadt.de/data/from_games/)放置于`./data/GTA5/` 路径下。

- 下载[CitysCapes数据集](https://www.cityscapes-dataset.com/)放置于`./data/Cityscapes/` 路径下。

**GTA5**：

数据集包括了24966张尺寸为1914×1052的图像。这些图像来源于基于洛杉矶真实地貌开发的GTA5游戏场景，并且最终经过人工标注而获得。图像中的场景和Cityscapes数据集中的场景高度相似，并且在图像标注上和Cityscapes数据集共享19个相同的类别。在本次复现中，我们使用了所有的GTA5数据集图像进行模型训练。

**Cityscapes**：

Cityscapes数据集包含了来自50个不同城市中驾驶场景的5000张图像，其中2957张图像用作训练，500张用作验证，1525张用作测试，尺寸为1024×2048。该数据集提供fine和coarse两种评测标准，前者提供5000张精标注图像，后者额外提供了20000张粗标注图像。

**数据集目录树**：
```text
data  
╠══Cityscapes  
║   ╠══gtFine  
║   ║   ╠══test  
║   ║   ╠══train  
║   ║   ╚══val  
║   ╚══leftimg8bit  
║       ╠══test  
║       ╠══train  
║       ╚══val 
╚══GTA5  
    ╠══images  
    ╚══labels  
```

## 指标描述

语义分割任务会给每一个像素赋予其对应的类别，最终所有的类别预测都对应图像中的若干个区域。对于单一类别而言，类别预测区域和正确区域的交并比即该类的IoU指标。将所有类别的IoU值进行平均化处理，就可以得到最终图像预测的mIoU指标，该指标表示了模型最终的语义分割精度情况，其指标数值越高表明语义分割所达到的精度越高。

数学形式表达如式（1）所示，$i$其中表示对应的类别，$predict$表示对应类别的预测区域，$target$表示对应类别的真实区域：

$$
\begin{aligned}
IoU_i &= \frac{predict_i \cap target_i}{predict_i \cup target_i} \\
mIoU &= \frac{1}{N}\sum_{i=1}^N IoU_i
\end{aligned}
$$

## 性能描述

| 参数          | Ascend                                                         | 
|-------------|----------------------------------------------------------------|
| 模型版本        | SND                                                            |
| 资源          |                                                                | 
| 上传日期        |                                                                |  
| MindSpore版本 | 2.2.14                                                         | 
| 数据集         | GTA5、Cityscapes                                                | 
| 训练参数        |                                                                | 
| 优化器         | Momentum                                                       | 
| 损失函数        | Softmax交叉熵                                                     | 
| 速度          | xxx毫秒/步;                                                       | 
| 总时长         | xxx分钟;                                                         |
| 推理模型        | [模型](https://pan.baidu.com/s/1LTtrg2KGcgLHB9EawCPiQg?pwd=nwkj) |                 |

## 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 项目目录结构

## 训练脚本

- 所有的训练脚本都保存在train.sh文件当中，如果需要可以自行查阅，并且选取进行修改。

- 所有的预训练模型可以通过[网盘链接](https://pan.baidu.com/s/1LTtrg2KGcgLHB9EawCPiQg?pwd=nwkj)直接进行下载。

- 下载后放置于`checkpoint`文件夹下

- 相关参数配置见`snd_config.yaml`

```shell
bash scripts/train.sh [DEVICE_ID] 
```

## 推理脚本

```shell
bash scripts/test.sh [DEVICE_ID] [MODEL_PATH]
```



