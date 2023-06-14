# FasterRCNN

***

论文：[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

标签： 目标检测

***

## 模型简介

Faster R-CNN是一个两阶段目标检测网络，该网络采用RPN，可以与检测网络共享整个图像的卷积特征，可以几乎无代价地进行区域候选计算。整个网络通过共享卷积特征，进一步将RPN和Fast R-CNN合并为一个网络。

<img src="https://user-images.githubusercontent.com/40661020/143881188-ab87720f-5059-4b4e-a928-b540fb8fb84d.png"/>

## 数据集

使用的数据集：[COCO2017](https://cocodataset.org/)

在官网下载完成后将数据集组织成如下结构：

```text
└── coco2017
     ├── train2017
     │    ├── 000000000009.jpg
     │    ├── 000000000025.jpg
     │    ├── ...
     ├── test2017
     │    ├── 000000000001.jpg
     │    ├── 000000058136.jpg
     │    ├── ...
     ├── val2017
     │    ├── 000000000139.jpg
     │    ├── 000000057027.jpg
     │    ├── ...
     └── annotation
          ├── captions_train2017.json
          ├── captions_val2017.json
          ├── instances_train2017.json
          ├── instances_val2017.json
          ├── person_keypoints_train2017.json
          └── person_keypoints_val2017.json
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

| Model | pretrained Model | config | Device Num | Epoch | mAP(0.5~0.95) | CheckPoint | Graph Train Log |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| FasterRCNN R50-FPN | [R50](https://github.com/mindspore-lab/mindcv) | [cfg](config/faster_rcnn/faster_rcnn_resnet50_fpn_1x.yml) | 8 | 12 | 37.0 | [download]() | [download]() |

## 快速入门

修改 `cconfig/coco.yml` 中 coco 数据集的路径：

```text
dataset_dir: "your cityscapes"
```

### 单卡训练

```shell
python train.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True

# 例如，单卡训练 FasterRCNN R50-FPN
python train.py --config config/faster_rcnn/faster_rcnn_resnet50_fpn_1x.yml --device_target Ascend --mix True
```

### 多卡训练

```shell
mpirun --allow-run-as-root -n [DEVICE_NUM] --merge-stderr-to-stdout \
    python train.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True

# 例如，8卡训练 FasterRCNN R50-FPN
mpirun --allow-run-as-root -n 8 --merge-stderr-to-stdout 、
    python train.py --config config/faster_rcnn/faster_rcnn_resnet50_fpn_1x.yml --device_target Ascend --mix True
```

### 推理评估

```shell
python eval.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True --checkpoint_path [CHECKPOINT_PATH]

# 例如，单卡推理 FasterRCNN R50-FPN
python eval.py --config config/faster_rcnn/faster_rcnn_resnet50_fpn_1x.yml --device_target Ascend --mix True --checkpoint_path output/checkpoint/FasterRCNN_det_resnet50_epoch12_rank0.ckpt

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

Q: 遇到内存不够或者线程数过多的WARNING怎么办?

A: 调整config文件中的`num_workers`： 并行数，`prefetch_size`：缓存队列长度， `max_rowsize`：一条数据在MB里最大内存占用，batchsize 16 最小设9。
一般CPU占用过多需要减小`num_workers`；内存占用过多需要减小`num_workers`，`prefetch_size`和`max_rowsize`。

Q: GPU环境上loss不收敛怎么办?

A: 将`config`文件中的`mix`改成`False`。
