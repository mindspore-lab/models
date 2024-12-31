# MMoE描述

## 概述

为了解决任务之间相关性降低导致模型效果下降的问题，在MoE的基础上进行改进，提出了MMoE。MMoE为每一个task设置一个gate，用这些gate控制不同任务不同专家的权重。

## 论文

1. [论文](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-): modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-

# 数据集

- 数据集1：[Census-income](http://github.com/drawbridge/keras-mmoe)
- 数据集2：[Adult](https://archive.ics.uci.edu/dataset/2/adult)
- 数据集3：[Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

注：所有的数据在需要先在data.py中处理成mindrecord格式再进行训练和推理。

**重要提醒： 以Census-income为例介绍数据预处理的细节：**
![image-20241111173028353](C:\Users\ROG\AppData\Roaming\Typora\typora-user-images\image-20241111173028353.png)

上图为Census-income所对应的data.py文件，其中column_names为原始数据中所有的列名，label_columns为要预测的标签（多任务学习），categorical_columns为真正用于训练的特征。

**注意：**

1. categorical_columns中的列必须都是离散的特征，代码会对所有的用于训练的特征进行one-hot编码；

2. 训练集和测试集中的所有离散特征的可能的取值都必须完全一致；

3. data.py和default_config.yaml中的文件名也需要根据数据集的名称相应地更改：

   <img src="C:\Users\ROG\AppData\Roaming\Typora\typora-user-images\image-20241111185859021.png" alt="image-20241111185859021" style="zoom: 50%;" />

   <img src="C:\Users\ROG\AppData\Roaming\Typora\typora-user-images\image-20241111185958685.png" alt="image-20241111185958685" style="zoom:50%;" />

4. 需要计算所有特征的不同取值的个数num_features，并修改对应配置文件中的参数值：

   ![image-20241111173806548](C:\Users\ROG\AppData\Roaming\Typora\typora-user-images\image-20241111173806548.png)

注意以上细节后即可对数据预处理，数据预处理命令为：

```
python data.py --local_data_path  ./data  # 数据集的两个压缩包直接放在主目录的data文件夹下即可
```

在上传的项目中我们也给出了adult和Bank_Marketing数据集的更改示例，可供参考：

![image-20241111190706134](C:\Users\ROG\AppData\Roaming\Typora\typora-user-images\image-20241111190706134.png)

# 环境要求

- 硬件
    - 准备Ascend处理器搭建硬件环境
- 框架
    - [MindSpore](https://www.mindspore.cn/install/) 1.8
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html)

# 训练与评估

在预处理数据后，您可以按照如下步骤进行训练和评估：

- 使用Ascend处理器

```Shell
# 单机训练(Ascend)
bash run_standalone_train_ascend.sh [DATA_PATH] [DEVICE_ID] [CKPT_PATH] [CONFIG_FILE]
[DATA_PATH]是预处理后的训练数据集的路径。
[CKPT_PATH]是要将ckpt保存的位置。
[DEVICE_ID]为执行train.py的ID号。
[CONFIG_FILE]是模型及运行的整体参数（yaml文件）。

# 运行评估示例（Ascend）
bash run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID] [CONFIG_FILE]
[DATA_PATH]是预处理后的测试数据集的路径。
[CKPT_PATH]是保存ckpt的位置。
[DEVICE_ID]为执行eval.py的ID号。
[CONFIG_FILE]是模型及运行的整体参数（yaml文件）。
```

# 脚本说明

## 脚本及样例代码

```text
└──mmoe
  ├── README_CN.md
  ├── ascend310_infer
    ├── inc
      ├── util.h
    ├── src
      ├── build.sh
      ├── CMakeList.txt
      ├── main.cc
      ├── utils.cc
  ├── scripts
    ├── run_distribute_ascend.sh            # 启动Ascend分布式训练（8卡）
    ├── run_standalone_eval_ascend.sh       # 启动Ascend910评估
    ├── run_standalone_eval_gpu.sh          # 启动GPU评估
    ├── run_infer_310.sh                    # 启动Ascend310评估
    ├── run_standalone_train_ascend.sh      # 启动Ascend单机训练（单卡）
    └── run_standalone_train_gpu.sh         # 启动GPU单机训练（单卡）
  ├── src
    ├── model_utils
        ├── config.py                        # 参数配置
        ├── device_adapter.py                # 适配云上或线下
        ├── local_adapter.py                 # 线下配置
        ├── moxing_adapter.py                # 云上配置
    ├── callback.py                          # 训练过程中进行评估的回调  
    ├── data.py                              # 数据预处理
    ├── load_dataset.py                      # 加载处理好的mindrecord格式数据
    ├── get_lr.py                            # 生成每个步骤的学习率
    ├── mmoe.py                              # 模型整体架构
    └── mmoe_utils.py                        # 每一层架构
  ├── eval.py                                # 910评估网络
  ├── default_config.yaml                    # 默认的参数配置
  ├── default_config_cpu.yaml                # 针对CPU环境默认的参数配置
  ├── default_config_gpu.yaml                # 针对GPU环境默认的参数配置
  ├── export.py                              # 910导出网络
  ├── fine_tune.py                           # CPU训练网络
  ├── postprocess.py                         # 310推理精度计算
  ├── preprocess.py                          # 310推理前数据处理
  └── train.py                               # 910训练网络
```

# 脚本参数

在default_config.yaml中可以同时配置训练参数和评估参数。

- 配置MMoE和Census-income数据集。

```Python
"num_features":499,                # 每一条数据的特征数
"num_experts":8,                   # 专家数
"units":4,                         # 每一层的unit数
"batch_size":32,                   # 输入张量的批次大小
"epoch_size":100,                  # 训练周期大小
"learning_rate":0.001,             # 初始学习率
"save_checkpoint":True,            # 是否保存检查点
"save_checkpoint_epochs":1,        # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,          # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":5,                 # 热身周期
```

- CPU环境下参数设置

```Python
"num_features":499,                # 每一条数据的特征数
"num_experts":8,                   # 专家数
"units":4,                         # 每一层的unit数
"batch_size":32,                   # 输入张量的批次大小
"epoch_size":10,                  # 训练周期大小
"learning_rate":0.0001,             # 初始学习率
"save_checkpoint":True,            # 是否保存检查点
"save_checkpoint_epochs":1,        # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,          # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":5,                 # 热身周期
```

# 结果

推理结果保存在脚本执行的当前路径，
您可以在当前文件夹中acc.log查看推理精度，在time_Result中查看推理时间。

# 随机情况说明

train.py中使用随机种子
