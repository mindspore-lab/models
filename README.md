<<<<<<< HEAD
# ![MindSpore Logo](https://gitee.com/mindspore/mindspore/raw/master/docs/MindSpore-logo.png)

## Welcome to the MindSpore Model Zoo
| Directory | Description |
| :-- | :-- |
| [official](official)    | • SOTA models support list <br/>• Maintained by MindSpore Team        |  

### MindSpore Lab:
| model toolkit | introduction |  
|:--:|:-- |
| mindformers | • transformers-like large models, includes llama/qwen etc. <br/> • https://github.com/mindspore-lab/mindformers      |
| mindone     | • text to image/video/audio models，includes sd/sora/dit etc.   <br/> • https://github.com/mindspore-lab/mindone     | 
| mindcv      | • cv models includes vgg/resnet/vit etc.         <br/> •  https://github.com/mindspore-lab/mindcv                    | 
| mindnlp     | • nlp models includes bert/roberta etc.      <br/> • https://github.com/mindspore-lab/mindnlp                        | 
| mindaudio   | • audio models includes deepspeech/conformer etc. <br/> • https://github.com/mindspore-lab/mindaudio                 |
| mindocr     | • ocr models includes dbnet/crnn etc.          <br/> • https://github.com/mindspore-lab/mindocr                      | 
| mindyolo    | • yolo models collection includes yolov3~v10 etc.       <br/> • https://github.com/mindspore-lab/mindyolo            |

## Disclaimers

Mindspore only provides scripts that downloads and preprocesses public datasets. We do not own these datasets and are not responsible for their quality or maintenance. Please make sure you have permission to use the dataset under the dataset’s license. The models trained on these dataset are for non-commercial research and educational purpose only.

To dataset owners: we will remove or update all public content upon request if you don’t want your dataset included on Mindspore, or wish to update it in any way. Please contact us through a Github/Gitee issue. Your understanding and contribution to this community is greatly appreciated.

MindSpore is Apache 2.0 licensed. Please see the LICENSE file.

## License

[Apache License 2.0](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)
=======
# CIFAR-10 分类：MAAM 模型训练

## 概述
本项目实现了基于 MAAM（多代理聚合模块）的神经网络，用于对 CIFAR-10 数据集进行分类。模型通过结合多代理特征图并使用 Agent Attention 机制来聚合信息，并最终进行图像分类。数据加载和增强采用 MindSpore 框架，训练和评估结果存储在文本文件中。

## 数据集
该模型使用 CIFAR-10 数据集，数据集包含 60,000 张 32x32 彩色图像，分为 10 类。数据集划分为训练集（50,000 图像）和测试集（10,000 图像）。数据增强包括随机裁剪、随机水平翻转和颜色归一化等操作。

## 网络结构
模型包括以下几个模块：
1. **AgentBlock**：每个代理模块使用卷积层提取局部特征。
2. **AgentAttentionFusion**：为每个代理分配一个可学习的权重，并通过加权融合不同代理的特征。
3. **MAAModule**：将多个代理特征图组合并使用 Agent Attention 机制融合。
4. **MAAMNetwork**：包含 MAAM 模块和分类头，最终输出分类结果。

## 训练步骤
1. **数据加载**：从指定路径加载 CIFAR-10 数据集，并进行数据增强处理。
2. **模型训练**：使用 Adam 优化器，进行 50 个 epoch 的训练，并在每个 epoch 结束后计算训练集和测试集的评估指标，包括准确率、精确度、召回率和 F1 分数。
3. **保存训练结果**：训练过程中，所有指标会写入到 `training_results(main+cnn).txt` 文件中。

## 训练结果
### 输出内容
每个 epoch 结束后，训练过程和验证过程的结果会分别输出并保存。输出内容包括：
- **训练集**：
  - 损失值（Loss）
  - 准确率（Accuracy）
  - 精确度（Precision）
  - 召回率（Recall）
  - F1 分数（F1-Score）
  - 训练时间（Training Time）
  
- **验证集**：
  - 测试准确率（Test Accuracy）
  - 测试精确度（Test Precision）
  - 测试召回率（Test Recall）
  - 测试 F1 分数（Test F1-Score）
>>>>>>> maam/main
