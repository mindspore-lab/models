# OCRNet

***

论文：[Object-Contextual Representations for Semantic Segmentation](https://arxiv.org/pdf/1909.11065)

标签：语义分割

***

## 模型简介

OCRNet是由微软亚研院和中科院计算所提出的语义分割网络。OCRNet使用了一种新的物体上下文信息——在构建上下文信息时显式地增强了来自于同一类物体的像素的贡献，并在2019年7月和2020年1月的 Cityscapes leaderboard提交结果中都取得了语义分割任务第一名的成绩。相关工作“Object-Contextual Representations for Semantic Segmentation”已经被 ECCV 2020 收录。

<img src="https://user-images.githubusercontent.com/24582831/142902197-b06b1e04-57ab-44ac-adc8-cea6695bb236.png">

## 数据集

使用的数据集：[Cityscapes](https://www.cityscapes-dataset.com/)

在官网下载完成后将数据集组织成如下结构：

```text
└── Cityscapes
     ├── leftImg8bit
     |    ├── train
     |    |    ├── aachen
     |    |    |    ├── aachen_000000_000019_leftImg8bit.png
     |    |    |    ├── aachen_000001_000019_leftImg8bit.png
     |    |    |    ├── ...
     |    |    ├── bochum
     |    |    |    ├── ...
     |    |    ├── ...
     |    ├── test
     |    |    ├── ...
     |    ├── val
     |    |    ├── ...
     └── gtFine
          ├── train
          |    ├── aachen
          |    |    ├── aachen_000000_000019_gtFine_color.png
          |    |    ├── aachen_000000_000019_gtFine_instanceIds.png
          |    |    ├── aachen_000000_000019_gtFine_labelIds.png
          |    |    ├── aachen_000000_000019_gtFine_polygons.json
          |    |    ├── aachen_000001_000019_gtFine_color.png
          |    |    ├── aachen_000001_000019_gtFine_instanceIds.png
          |    |    ├── aachen_000001_000019_gtFine_labelIds.png
          |    |    ├── aachen_000001_000019_gtFine_polygons.json
          |    |    ├── ...
          |    ├── bochum
          |    |    ├── ...
          |    ├── ...
          ├── test
          |    ├── ...
          └── val
               ├── ...
```

## 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。参考[MindSpore](https://www.mindspore.cn/install/en)安装运行环境
- 版本依赖 MindSpore >= 2.0

```shell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## BenchMark

### 精度

| Model | pretrained Model | config  | Device Num | Step | Test Size | mIoU | mIou(ms) | CheckPoint | Graph Train Log |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| OCRNet-hrw48 | [hrw48](https://github.com/mindspore-lab/mindcv) | [cfg](config/ocrnet/config_ocrnet_hrw48_16k.yml) | 8 | 16k | [1024, 2048] |  |  | [download]() | [download]() |
| OCRNet-hrw32 | [hrw32](https://github.com/mindspore-lab/mindcv) | [cfg](config/ocrnet/config_ocrnet_hrw32_16k.yml) | 8 | 16k | [1024, 2048] |  |  | [download]() | [download]() |

### 性能

| device | Model     | dataset   | Params(M) | Graph train 8P bs=2 FPS |
| ------ | --------- | --------- | --------- | ------- |
| Ascend | OCRNet-hrw48 | Cityscapes |  M  |  |
| Ascend | OCRNet-hrw32 | Cityscapes |  M  |  |

以上数据是在

Ascend 910 32G 8卡；系统： Euler2.8；内存：756 G；x86 96核 CPU；

机器上进行的实验。

## 快速入门

修改 `config/cityscapes.yml` 中 Cityscapes 数据集的路径：

```text
dataset_dir: "your cityscapes"
```

### 单卡训练：

```shell
python train.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True

# 例如，单卡训练 OCRNet-hrw48
python train.py --config config/ocrnet/config_ocrnet_hrw32_16k.yml --device_target Ascend --mix True
```

### 多卡训练：

```shell
mpirun --allow-run-as-root -n [DEVICE_NUM] --merge-stderr-to-stdout \
    python train.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True

# 例如，8卡训练 OCRNet-hrw48
mpirun --allow-run-as-root -n 8 --merge-stderr-to-stdout 、
    python train.py --config config/ocrnet/config_ocrnet_hrw32_16k.yml --device_target Ascend --mix True
```

### 推理评估：

```shell
python eval.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True --checkpoint_path [CHECKPOINT_PATH]

# 例如，单卡推理 OCRNet-hrw48
python eval.py --config config/ocrnet/config_ocrnet_hrw32_16k.yml --device_target Ascend --mix True --checkpoint_path output/checkpoint/OCRNet_hrnet_seg_w48_160000_rank0.ckpt

# 也可以多卡推理，例如
mpirun --allow-run-as-root -n [DEVICE_NUM] --merge-stderr-to-stdout 、
    python eval.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True --checkpoint_path [CHECKPOINT_PATH]
```

### 断点训练

如果想使用断点训练功能，在启动时添加 resume_ckpt 训练参数即可， 如：

```shell
python train.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True --resume_ckpt [CHECKPOINT_PATH]
```

### ModelArts 上训练

1. 在config文件中配置 ModelArts 参数：

- 设置 enable_modelarts=True
- 设置OBS数据集路径 data_url: <数据集在OBS中的路径>
- 设置OBS训练回传路径 train_url: <输出文件在OBS中的路径>

2. 按照[ModelArts教程](https://support.huaweicloud.com/modelarts/index.html)执行训练。

## 免责声明

models仅提供下载和预处理公共数据集的脚本。我们不拥有这些数据集，也不对它们的质量负责或维护。请确保您具有在数据集许可下使用该数据集的权限。在这些数据集上训练的模型仅用于非商业研究和教学目的。

致数据集拥有者：如果您不希望将数据集包含在MindSpore models中，或者希望以任何方式对其进行更新，我们将根据要求删除或更新所有公共内容。请通过 Gitee 与我们联系。非常感谢您对这个社区的理解和贡献。

## FAQ

优先参考 [Models FAQ](https://gitee.com/mindspore/models#FAQ) 来查找一些常见的公共问题。

Q: Acend上单step时间与README中数据相差很大?

A: loss中使用了`nn.CrossEntropyLoss`，其中`logsumexp`使用了`input.max()`操作，这个操作底层使用的ArgMaxWithValue算子在大shape下会很慢，
最新版本已修改成ReduceMax算子，之前的版本可手动修改MindSpore内代码，可参考[PR 55127](https://gitee.com/mindspore/mindspore/pulls/55127)。

Q: 遇到内存不够或者线程数过多的WARNING怎么办?

A: 调整config文件中的`num_workers`： 并行数，`prefetch_size`：缓存队列长度， `max_rowsize`：一条数据在MB里最大内存占用，batchsize 16 最小设9。
一般CPU占用过多需要减小`num_workers`；内存占用过多需要减小`num_workers`，`prefetch_size`和`max_rowsize`。

Q: GPU环境上loss不收敛怎么办?

A: 将`config`文件中的`mix`改成`False`。
