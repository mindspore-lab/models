 # **项目描述项目描述**

本项目旨在利用[MindSpore](https://www.mindspore.cn/install/en)框架对[CAMP](https://arxiv.org/abs/1909.05506)模型进行复现。

## CAMP描述

[CAMP](https://arxiv.org/abs/1909.05506)（跨模态自适应消息传递模型）自适应地控制跨模态信息流，不仅考虑了全面和细粒度的跨模态交互，而且还通过自适应门控方案正确处理负对和不相关信息。此外，[CAMP](https://arxiv.org/abs/1909.05506)（没有传统的文本图像匹配联合嵌入方法，而是基于融合特征推断匹配分数，并提出了一种最难的负二元交叉熵损失进行训练。

## 模型结构描述

- [CAMP](https://arxiv.org/abs/1909.05506)基于跨模态注意来考虑跨模态信息聚合，考虑将原始信息与其他模态传递的聚合信息融合，因为融合负（不匹配）对使得很难找到信息对齐，所以最后引入了跨模态门控融合模块来自适应地控制对齐和不对齐信息的融合。
- 结构如图[figure 1](./figure/CAMP_pipline.bmp)所示。

## 数据集描述


- 本次复现在一个数据集上完成，是Flickr30k数据集。
- Flickr30k数据集总共包含 31000 张图像。通常，每个图 像都是对应 5 个文本描述。使用 1000 张图像进行测试，另外 1000 张有效的图像，其余用于训练。Flickr30K 的图像特征是从原始 Flickr30K 图像中提取的，使用[自下而上的注意力模型](https://github.com/peteanderson80/bottom-up-attention)进行特征提取的。再现论文中实验所需的所有数据，包括图像特征和词汇，都可以从以下位置下载：https://www.kaggle.com/datasets/kuanghueilee/scan-features

**数据集目录**：

data  
╠══f30k_precomp  
║      ╚══test_ids.txt  
║      ╚══test_tags.txt                  
║      ╚══test_caps.txt          
║      ╚══test_ims.npy         
║      ╚══dev_ids.txt        
║      ╚══dev_tags.txt  
║      ╚══dev_caps.txt  
║      ╚══dev_ims.npy          
║      ╚══train_ids.txt         
║      ╚══train_tags.txt        
║      ╚══train_caps.txt       
║      ╚══train_ims.npy  
╚══vocab  
        ╠══f30k_precomp_vocab.json  

## 指标描述


| 参数          | Ascend   | GPU      |
| ------------- |----------|----------|
| 模型版本      | CAMP | CAMP |
| MindSpore版本 | 1.10.0  | 1.10.0  |
| 数据集        | Flickr30k | Flickr30k |
| 优化器        | SGD | SGD |
| 损失函数      | 对比损失 | 对比损失 |
| 速度          | 890毫秒/步; | xxx毫秒/步; |
| 总时长        | 分钟; | xxx分钟;   |



## 项目目录结构

```
|-- data
|   |--   ...  #原始数据，形式如上
|-- pretrain_weight
	|--   checkpoint_110.pth.tar   # 预训练模型
	|--   ...   # 训练时会产生其它权重
|-- logs
|   |--   ...  # 存训练模型参数
|-- src
|   |-- __init__.py
|   |-- rnn_utils.py
|   |-- data.py
|   |-- rnn_cells.py
|   |-- model.py
|   |-- evaluation.py
|   |-- pth2ckpt.py.py
|   |-- rnns.py
|   |-- fusion_module.py
|-- figure
|   |-- CAMP_pipline.png
|-- vocab
|   |-- f30k_precomp_vocab.pkl
|   |-- f30k_precomp_vocab_1.pkl
|   |-- f30k_vocab.pkl
|-- run_test.sh
|-- run_train.sh
|-- config.yaml
|-- test.py
|-- vocab.py
|-- train.py
|-- README_CN.md
```

### 训练
首先下载预训练权重[checkpoint_110.pth.tar](https://drive.google.com/drive/folders/1o8rUv78uS_aX4P1hMPELl53cxnZ8UqiF?usp=sharing)
```
sh run_train.sh
```

### 测试

```
sh run_test.sh
```

