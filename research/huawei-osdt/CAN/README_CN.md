# 手写数学公式识别算法-CAN

- [手写数学公式识别算法-CAN](#手写数学公式识别算法-can)
  - [1. 算法简介](#1-算法简介)
  - [2. 环境配置](#2-环境配置)
  - [3. 模型推理、评估、训练](#3-模型推理评估训练)
    - [3.1 推理](#31-推理)
    - [3.2 数据集准备](#32-数据集准备)
      - [训练集准备](#训练集准备)
      - [验证集准备](#验证集准备)
    - [3.3 训练](#33-训练)
    - [3.4 评估](#34-评估)
  - [4. FAQ](#4-faq)
  - [引用](#引用)
  - [参考文献](#参考文献)



## 1. 算法简介
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->
> [CAN: When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition](https://arxiv.org/pdf/2207.11463.pdf)

CAN是具有一个弱监督计数模块的注意力机制编码器-解码器手写数学公式识别算法。本文作者通过对现有的大部分手写数学公式识别算法研究，发现其基本采用基于注意力机制的编码器-解码器结构。该结构可使模型在识别每一个符号时，注意到图像中该符号对应的位置区域，在识别常规文本时，注意力的移动规律比较单一（通常为从左至右或从右至左），该机制在此场景下可靠性较高。然而在识别数学公式时，注意力在图像中的移动具有更多的可能性。因此，模型在解码较复杂的数学公式时，容易出现注意力不准确的现象，导致重复识别某符号或者是漏识别某符号。

针对于此，作者设计了一个弱监督计数模块，该模块可以在没有符号级位置注释的情况下预测每个符号类的数量，然后将其插入到典型的基于注意的HMER编解码器模型中。这种做法主要基于以下两方面的考虑：1、符号计数可以隐式地提供符号位置信息，这种位置信息可以使得注意力更加准确。2、符号计数结果可以作为额外的全局信息来提升公式识别的准确率。

<p align="center">
  <img src="https://raw.githubusercontent.com/zhangjunlongtech/Material/refs/heads/main/CAN/miss_word.png" width=640 />
</p>
<p align="center">
  <em> 手写数学公式识别算法对比 [<a href="#参考文献">1</a>] </em>
</p>

CAN模型由主干特征提取网络、多尺度计数模块（MSCM）和结合计数的注意力解码器（CCAD）构成。主干特征提取通过采用DenseNet得到特征图，并将特征图输入MSCM，得到一个计数向量（Counting Vector），该计数向量的维度为1*C，C即公式词表大小，然后把这个计数向量和特征图一起输入到CCAD中，最终输出公式的latex。

<p align="center">
  <img src="https://raw.githubusercontent.com/zhangjunlongtech/Material/refs/heads/main/CAN/total_process.png" width=640 />
</p>
<p align="center">
  <em> 整体模型结构 [<a href="#参考文献">1</a>] </em>
</p>

多尺度计数模MSCM块旨在预测每个符号类别的数量，其由多尺度特征提取、通道注意力和池化算子组成。由于书写习惯的不同，公式图像通常包含各种大小的符号。单一卷积核大小无法有效处理尺度变化。为此，首先利用了两个并行卷积分支通过使用不同的内核大小（设置为 3×3 和 5×5）来提取多尺度特征。在卷积层之后，采用通道注意力来进一步增强特征信息。

<p align="center">
  <img src="https://raw.githubusercontent.com/zhangjunlongtech/Material/refs/heads/main/CAN/MSCM.png" width=640 />
</p>
<p align="center">
  <em> MSCM多尺度计数模块 [<a href="#参考文献">1</a>] </em>
</p>

结合计数的注意力解码器：为了加强模型对于空间位置的感知，使用位置编码表征特征图中不同空间位置。另外，不同于之前大部分公式识别方法只使用局部特征进行符号预测的做法，在进行符号类别预测时引入符号计数结果作为额外的全局信息来提升识别准确率。

<p align="center">
  <img src="https://raw.githubusercontent.com/zhangjunlongtech/Material/refs/heads/main/CAN/CCAD.png" width=640 />
</p>
<p align="center">
  <em> 结合计数的注意力解码器CCAD [<a href="#参考文献">1</a>] </em>
</p>

<a name="model"></a>
`CAN`使用CROHME手写公式数据集进行训练，在对应测试集上的精度如下：

|模型    |骨干网络|配置文件|ExpRate|下载链接|
| ----- | ----- | ----- | ----- | ----- |
|CAN|DenseNet|[rec_d28_can.yml](./configs/can_d28.yaml)|52.84%|[训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_d28_can_train.tar)|

<a name="2"></a>
## 2. 环境配置
请先安装 `MindSpore=2.4.0`，并进入项目目录执行 `pip install -e .`


<a name="3"></a>
## 3. 模型推理、评估、训练

<a name="3-1"></a>
### 3.1 推理

首先准备好模型权重文件，这里以项目提供的最佳权重文件为例 ([权重文件下载地址](https://download-mindspore.osinfra.cn/model_zoo/research/cv/can/))，执行如下命令进行推理：

```shell

Python  /tools/predict_can.py   --image_dir {path_to_img} \
						        --rec_algorithm CAN \
						        --rec_model_dir {path_to_ckpt} \
						        --rec_char_dict_path {path_to_dict} \

```
**注意：**
- 其中`--image_dir`为图像地址，`rec_model_dir`为权重文件地址，`rec_char_dict_path`为识别字典地址，在进行模型推理时需要根据实际地址修改
- 需要注意预测图像为**黑底白字**，即手写公式部分为白色，背景为黑色的图片。

![测试图片样例](https://raw.githubusercontent.com/zhangjunlongtech/Material/refs/heads/main/CAN/101_user0.jpg)

执行命令后，上面图像的预测结果（识别的文本）会打印到屏幕上，示例如下：
```shell
All rec res: ['S = ( \\sum _ { i = 1 } ^ { n } \\theta _ { i } - ( n - 2 ) \\pi ) r ^ { 2 }']
```

<a name="3-2"></a>

### 3.2 数据集准备

本模型提供的数据集，即[`CROHME数据集`](https://paddleocr.bj.bcebos.com/dataset/CROHME.tar)将手写公式存储为黑底白字的格式，若您自行准备的数据集与之相反，训练前请统一处理数据集。


#### 训练集准备
请将所有训练图片置入同一文件夹，并在上层路径指定一个txt文件用来标注所有训练图片名和对应标签。txt文件例子如下

```
# 文件名	# 对应标签
word_421.png	k ^ { 3 } + 1 4 k ^ { 2 } - 1 3 2 k + 1 7 8 9
word_1657.png	 x _ { x } ^ { x } + y _ { y } ^ { y } + z _ { z } ^ { z } - x - y - z
word_1814.png	\sqrt { a } = 2 ^ { - n } \sqrt { 4 ^ { n } a }
```
*注意*：请将图片名和标签以 \tab 作为分隔，避免使用空格或其他分隔符。

最终训练集存放会是以下形式：

```
|-data
    |- gt_training.txt
    |- training
        |- word_001.png
        |- word_002.jpg
        |- word_003.jpg
        | ...
```

#### 验证集准备
同样，请将所有验证图片置入同一文件夹，并在上层路径指定一个txt文件用来标注所有验证图片名和对应标签。最终验证集存放会是以下形式：

```
|-data
    |- gt_validation.txt
    |- validation
        |- word_001.png
        |- word_002.jpg
        |- word_003.jpg
        | ...
```

### 3.3 训练

在完成数据准备后，便可以启动训练，训练命令如下：
```shell
python tools/train.py --config configs/rec_d28_can.yaml
```
多卡训练命令如下：
```shell
mpirun --allow-run-as-root -n {card_nums} python tools/train.py --config configs/can_d28.yaml
```
**注意：**
- 需要进入到`configs/rec_d28_can.yaml`配置文件当中，配置`character_dict_path`、`ckpt_load_path`为当前地址
- 需要在配置文件`model-head`条目下，将`is_train`调整为`True`
- 需要在配置文件`train-dataset`条目下，将`dataset_root`、`data_dir`、`label_file`调整为训练数据集实际地址
- 命令行中`eval.py`以及`rec_d28_can.yaml`文件可替换为文件绝对地址


### 3.4 评估

对已训练完成的模型文件，使用如下命令进行评估：

```shell
python tools/eval.py --config configs/rec_d28_can.yaml
```
**注意：**
- 需要进入到`configs/rec_d28_can.yaml`配置文件当中，配置`character_dict_path`、`ckpt_load_path`为当前地址
- 需要在配置文件`model-head`条目下，将`is_train`调整为False
- 需要在配置文件`eval-dataset`条目下，将`dataset_root`、`data_dir`、`label_file`调整为评估数据集实际地址
- 命令行中`eval.py`以及`rec_d28_can.yaml`文件可替换为文件绝对地址

<a name="3-1"></a>




<a name="5"></a>
## 4. FAQ

1. CROHME数据集来自于[CAN源repo](https://github.com/LBH1024/CAN) 。

## 引用

```bibtex
@misc{https://doi.org/10.48550/arxiv.2207.11463,
  doi = {10.48550/ARXIV.2207.11463},
  url = {https://arxiv.org/abs/2207.11463},
  author = {Li, Bohan and Yuan, Ye and Liang, Dingkang and Liu, Xiao and Ji, Zhilong and Bai, Jinfeng and Liu, Wenyu and Bai, Xiang},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```


## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->
[1] Xiaoyu Yue, Zhanghui Kuang, Chenhao Lin, Hongbin Sun, Wayne Zhang. RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition. arXiv:2007.07542, ECCV'2020
