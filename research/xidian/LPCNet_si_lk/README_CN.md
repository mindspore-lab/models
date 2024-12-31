# 目录

- [目录](#目录)
    - [LPCNet 描述](#lpcnet描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
            - [训练](#训练)
        - [评估过程](#评估过程)
            - [评估](#评估)
    - [推理过程](#推理过程)
        - [生成网络输入数据](#生成网络输入数据)
        - [导出MindIR](#导出mindir)
        - [结果](#结果)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [评估性能](#评估性能)
        - [推理性能](#推理性能)

<!-- /TOC -->

## [LPCNet描述](#目录)

LPCNet是一种将线性预测与递归神经网络相结合的低比特率神经网络声码器。

[论文](https://jmvalin.ca/papers/lpcnet_codec.pdf): J.-M. Valin, J. Skoglund, A Real-Time Wideband Neural Vocoder at 1.6 kb/s Using LPCNet, Proc. INTERSPEECH, arxiv:1903.12087, 2019.

## [模型架构](#目录)

LPCNet分为两部分：帧率网络和采样率网络。帧率网络由两个卷积层和两个全连接层组成，从5帧上下文中提取特征。采样率网络由两个GRU层和一个双全连接层（在全连接层的基础上进行修改）组成，且第一个GRU层的权重进行了稀疏化。采样率网络获取帧率网络提取的特征，同时对当前时间步进行线性预测，对前一个时间步进行采样结束激励，并通过sigmoid函数和二叉树概率表示预测当前激励。

## [数据集](#目录)

使用的数据集：[si_lk](<https://www.openslr.org/resources/30/si_lk.tar.gz>)

- 数据集大小：699MB
    - 选择其中20条数据作为测试集，其余为训练集。
    
- 数据格式：wav文件
    - 注：需要将wav文件转换为flac格式，并且每个flac文件应该嵌套两层的文件夹（或修改脚本）

- 下载数据集（只需要.wav文件）。目录结构如下：

    ```mls_polish
    ├─COPYING
    ├─README
    ├─etc
    └─orig
      ├─arctic_a0001.wav
      ├─arctic_a0002.wav
      ├─arctic_a0003.wav
      ...
      └─arctic_b0539.wav
    ```
    
- 数据的结构如下（train-clean中给出了示例）：

    ![image-20241122214645331](C:\Users\ROG\AppData\Roaming\Typora\typora-user-images\image-20241122214645331.png)





## [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install) 2.2
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html)

## [训练与评估](#目录)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```bash
# 输入脚本目录，开始编译特征量化。
bash run_compile.sh

# 将.wav格式文件转化为.flac格式文件
bash convert_wav_to_flac.sh
# 示例 bash convert_wav_to_flac.sh [SOURCE_DIRECTORY] [DESTINATION_DIRECTORY]

# 生成训练数据（运行命令前需按照顺序安装SoX）。
bash run_process_train_data.sh [TRAIN_DATASET_PATH] [OUTPUT_PATH]
# 示例：bash run_process_train_data.sh ./data_path/train ~/dataset_path/training_dataset/

# 训练LPCNet
bash run_standalone_train_ascend.sh [PREPROCESSED_TRAINING_DATASET_PATH] [CHECKPOINT_SAVE_PATH]
# 示例: bash run_standalone_train_ascend.sh ~/dataset_path/training_dataset/ ./ckpt/

# 生成测试数据（从test中选择20个文件进行评估）。
bash run_process_eval_data.sh [EVAL_DATASET_PATH] [OUTPUT_PATH]
# 示例：bash run_process_eval_data.sh ./dataset_path/test ~/dataset_path/test_dataset/

# 评估LPCNet
bash run_eval_ascend.sh [TEST_DATASET_PATH] [OUTPUT_PATH] [CHECKPOINT_SAVE_PATH]
# 示例：bash run_eval_ascend.sh ~/dataset_path/test_features/ ./eval_results/ ./ckpt/lpcnet-4.ckpt
```
