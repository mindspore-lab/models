# FasterRCNN

***

Paper: [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

label: object detection

***

## Introduction

Faster R-CNN is a two-stage object detection network that uses RPN and can share the convolutional features of the entire image with the detection network, 
allowing for almost cost free region candidate calculations. 
The entire network further combines RPN and Fast R-CNN into one network by sharing convolutional features.

<img src="https://user-images.githubusercontent.com/40661020/143881188-ab87720f-5059-4b4e-a928-b540fb8fb84d.png"/>

## Dataset

Dataset: [COCO2017](https://cocodataset.org/)

After downloading on the official website, organize the dataset into the following structure:

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

## Environment Requirements

- Hardware（Ascend/GPU/CPU）
  - Prepare hardware environment with Ascend processor. Reference [MindSpot](https://www.mindspore.cn/install/en) Installation and operation environment
- Dependency: MindSpore >= 2.0

```shell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## BenchMark

### mAP

| Model | pretrained Model | config | Device Num | Epoch | mAP(0.5~0.95) | CheckPoint | Graph Train Log |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| FasterRCNN R50-FPN | [R50](https://github.com/mindspore-lab/mindcv) | [cfg](config/faster_rcnn/faster_rcnn_resnet50_fpn_1x.yml) | 8 | 12 | 37.0 | [download]() | [download]() |

## Quick Start

change path of the Coco dataset in:`cconfig/coco.yml`:

```text
dataset_dir: "your cityscapes"
```

### Standalone Training

```shell
python train.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True

# example FasterRCNN R50-FPN training on 1 device
python train.py --config config/faster_rcnn/faster_rcnn_resnet50_fpn_1x.yml --device_target Ascend --mix True
```

### Distribute Training

```shell
mpirun --allow-run-as-root -n [DEVICE_NUM] --merge-stderr-to-stdout \
    python train.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True

# example FasterRCNN R50-FPN training on 8 devices
mpirun --allow-run-as-root -n 8 --merge-stderr-to-stdout 、
    python train.py --config config/faster_rcnn/faster_rcnn_resnet50_fpn_1x.yml --device_target Ascend --mix True
```

### Evaluation：

```shell
python eval.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True --checkpoint_path [CHECKPOINT_PATH]

# example FasterRCNN R50-FPN evaluation on 1 device
python eval.py --config config/faster_rcnn/faster_rcnn_resnet50_fpn_1x.yml --device_target Ascend --mix True --checkpoint_path output/checkpoint/FasterRCNN_det_resnet50_epoch12_rank0.ckpt

# distribute evaluation
mpirun --allow-run-as-root -n [DEVICE_NUM] --merge-stderr-to-stdout 、
    python eval.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True --checkpoint_path [CHECKPOINT_PATH]
```

### Resume Training

If you want to use breakpoint training function, add resume at startup_ CKPT training parameters are sufficient, such as:

```shell
python train.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --mix True --resume_ckpt [CHECKPOINT_PATH]
```

### Training on ModelArts

1. Configure the ModelArts parameter in the config file:

- set enable_modelarts=True
- set OBS dataset path data_url: <the path of the dataset in OBS>
- set OBS output results path train_url: <The path of output results in OBS>

2. Refer to [ModelArts](https://support.huaweicloud.com/modelarts/index.html) start training.

## Disclaimers

Mindspore only provides scripts that downloads and preprocesses public datasets. We do not own these datasets and are not responsible for their quality or maintenance. Please make sure you have permission to use the dataset under the dataset’s license. The models trained on these dataset are for non-commercial research and educational purpose only.

To dataset owners: we will remove or update all public content upon request if you don’t want your dataset included on Mindspore, or wish to update it in any way. Please contact us through a Github/Gitee issue. Your understanding and contribution to this community is greatly appreciated.


## FAQ

Refer to [Models FAQ](https://gitee.com/mindspore/models#FAQ) Firstly to get some common FAQs.

Q: What to do when encountering a WARNING with insufficient memory or too many threads?

A: Modify `num_workers` which is parallel number, `prefetch_size` which is dataset cache queue length, `max_rowsize` which is Maximum memory usage of a piece of data in config`.

Generally, excessive CPU usage requires reduction `num_workers`; Excessive memory usage needs to be reduced `num_workers`, `prefetch_size` and `max_rowsize`.

Q: What should I do if loss does not converge on GPU?

A: set`mix` to `False`。
