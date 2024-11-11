 # 项目描述
 
本项目旨在利用[MindSpore](https://www.mindspore.cn/install/en)框架对[CLAN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Luo_Taking_a_Closer_Look_at_Domain_Shift_Category-Level_Adversaries_for_CVPR_2019_paper.pdf)模型进行复现。

## CLAN描述

传统的大多数域适应方法在特征空间上采用的是全局融合策略,这样会导致一个问题：全局对齐使得原本一部分已经对齐的类别的特征经过对抗训练之后又不对齐了。所以[CLAN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Luo_Taking_a_Closer_Look_at_Domain_Shift_Category-Level_Adversaries_for_CVPR_2019_paper.pdf)提出了一种类层级的对抗训练，对已经对齐的class不再进行域适应，只对那些还没有对齐的类域适应。

## 模型结构描述

- 模型主要有一个特征提取器、两个结构相同分类器和一个判别器组成。
- 结构如图[figure 1](./figure/clan_pipline.png)所示。

## 数据集描述


- 本次复现在两个数据集上完成，分别是GTA5数据集和Cityscapes数据集。
- 数据集包括了24966张尺寸为1914×1052的图像。这些图像来源于基于洛杉矶真实地貌开发的GTA5游戏场景，并且最终经过人工标注而获得。图像中的场景和Cityscapes数据集中的场景高度相似，并且在图像标注上和Cityscapes数据集共享19个相同的类别。在本次复现中，我们使用了所有的GTA5数据集图像进行模型训练。点击[链接]( https://download.visinf.tu-darmstadt.de/data/from_games/ )获取GTA5数据集。
- Cityscapes数据集包含了来自50个不同城市中驾驶场景的5000张图像，其中2957张图像用作训练，500张用作验证，1525张用作测试，尺寸为1024×2048。该数据集提供fine和coarse两种评测标准，前者提供5000张精标注图像，后者额外提供了20000张粗标注图像。点击[链接]( https://www.cityscapes-dataset.com/ )获取Cityscapes数据集。

**数据集目录**：

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


| 参数          | Ascend           |
|-------------|------------------|
| 模型版本        | CLAN             |
| MindSpore版本 | 2.2              |
| 数据集         | GTA5/Cityscapes  |
| 优化器         | Momentum         |
| 损失函数        | 多分类交叉熵           |
| 速度          | 10毫秒/步;          |
| 精度(MIoU)    | 42.25(原论文中为43.2) |


## 项目目录结构

```
.
|-- README_CN.md
|-- utils
|   |-- rank_0
|   |-- __init__.py
|   |-- loss.py
|   |-- softmax.py
|   `-- visual.py
|-- options
|   |-- __init__.py
|   |-- evaluate_options.py
|   `-- train_options.py
|-- model
|   |-- __init__.py
|   |-- CLAN_D.py
|   |-- CLAN_D.py
|   |-- DeepLab_resnet_pretrained_init-f81d91e8(1).ckpt
|   `-- Pretrain_DeeplabMulti.ckpt
|-- figure
|   |-- clan_pipline.png
|-- dataset
|   |-- __init__.py
|   |-- cityscapes_list
|   |-- gta5_dataset.py
|   `-- cityscapes_dataset.py
|-- CLAN_train_gta5_2_city.py
`-- CLAN_evaluate_city.py
```

### 训练
```
python CLAN_train_gta5_2_city.py 
```

### 测试
```
python CLAN_evaluate_city.py --restore-from  ./checkpoint/GTA5_best.pth 
```

点击[链接]( ./model/DeepLab_resnet_pretrained_init-f81d91e8(1).ckpt )获取预训练模型 