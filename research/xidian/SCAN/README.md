 # **项目描述项目描述**

本项目旨在利用[MindSpore](https://www.mindspore.cn/install/en)框架对[SCAN](https://openaccess.thecvf.com/content_ECCV_2018/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf)模型进行复现。

## SCAN描述

[SCAN](https://openaccess.thecvf.com/content_ECCV_2018/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf)研究了图像文本匹配的问题。推断物体或其他突出物（例如雪、天空、草坪）之间的潜在语义对齐以及句子中的对应词允许捕获视觉和语言之间的细粒度相互作用，并使图像-文本匹配更具可解释性。先前的工作要么简单地聚合所有可能的区域和单词对的相似性，而不以不同的方式关注或多或少重要的单词或区域，要么使用多步注意力过程来捕获有限数量的语义对齐，这不太可解释。在本文中，[SCAN](https://openaccess.thecvf.com/content_ECCV_2018/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf)提出了 Stacked Cross Attention，以使用句子中的图像区域和单词作为上下文来发现完整的潜在对齐，并推断图像-文本相似度。

## 模型结构描述

- 模型一开始利用自下而上的注意力来检测和将图像区域编码为特征，然后将句子中的单词以及句子上下文映射到公共空间，最后，使用 Stacked Cross Attention 通过对齐图像区域和单词特征来推断图像句子的相似性。
- 结构如图[figure 1](./figure/scan_pipline.bmp)所示。

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
| 模型版本      | SCAN | SCAN |
| MindSpore版本 | 1.10.0  | 1.10.0  |
| 数据集        | Flickr30k | Flickr30k |
| 优化器      | Adam        | SGD |
| 损失函数      | 对比损失 | 对比损失 |
| 速度          | 0.826毫秒/步; | xxx毫秒/步; |
| 总时长        | 629分钟; | xxx分钟;   |

## 项目目录结构

```
|-- data
|   |--   ...  # 原始数据，形式如上
|-- logs
    |-- checkpoint  # 存训练模型参数
		|-- ...   # 训练时会产生其它文件
|-- src
|   |-- __init__.py
|   |-- rnn_utils.py
|   |-- data.py
|   |-- rnn_cells.py
|   |-- model.py
|   |-- evaluation.py
|   |-- rnns.py
|-- figure
|   |-- SCAN_pipline.png
|-- run_test.sh
|-- run_train.sh
|-- run_val.sh
|-- val.py
|-- test.py
|-- vocab.py
|-- train.py
|-- README_CN.md
```

### 训练

```
训练
sh run_train.sh

验证
sh run_val.sh    # epochs的数值要求和run_train.sh中相同
```

### 测试

```
sh run_test.sh
```

