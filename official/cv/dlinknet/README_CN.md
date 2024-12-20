# D-LinkNet

## 模型简介

D-LinkNet模型基于LinkNet架构构建。实现方式见论文[D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.html)
在2018年的DeepGlobe道路提取挑战赛中，这一模型表现最好。该网络采用编码器-解码器结构、空洞卷积和预训练编码器进行道路提取任务。

## Requirements
 | mindspore | ascend driver | firmware | cann toolkit/kernel |
 |:---------:|:-------------:|:--------:|:-------------------:|
 | 2.3.1 | 24.1.rc2 | 7.3.0.2.220 | 8.0.RC2.beta1 |
 | 2.4.0 | 24.1.rc3 | 7.5.0.1.129 | 8.0.RC3.beta1 |
 ```shell
 pip install -r requirement.txt
 ```

## 预训练权重
- [Resnet34 Imagenet Checkpoint](https://download-mindspore.osinfra.cn/toolkits/mindcv/resnet/resnet34-f297d27e.ckpt)
- [Resnet50 Imagenet Checkpoint](https://download-mindspore.osinfra.cn/toolkits/mindcv/resnet/resnet50-f369a08d-910v2.ckpt)

## 数据集

[DeepGlobe Road Extraction Dataset](https://www.kaggle.com/balraj98/deepglobe-road-extraction-dataset)

- 说明：该数据集由6226个训练图像、1243个验证图像和1101个测试图像组成。每个图像的分辨率为1024×1024。数据集被表述为二分类分割问题，其中道路被标记为前景，而其他对象被标记为背景。
- 数据集大小：3.83 GB

    - 训练集：2.79 GB，6226张图像，包含对应的标签图像，原图像以`xxx_sat.jpg`命名，对应的标签图像则以`xxx_mask.png`命名。
    - 验证集：552 MB，1243张图像，不包含对应的标签图像，原图像以`xxx_sat.jpg`命名。
    - 测试集：511 MB，1101张图像，不包含对应的标签图像，原图像以`xxx_sat.jpg`命名。

- 注意：由于该数据集为比赛用数据集，验证集与测试集的标签图像不会公开，本人在采用了将训练集划出十分之一作为验证集验证模型训练精度的方法。
- 上面给出的数据集链接为上传到Kaggle社区中的，可以直接下载。

- 如果你不想自己划分训练集，你可以只下载 [这个百度网盘链接](https://pan.baidu.com/s/1DofqL6P13PEDGUvNMPo-1Q?pwd=5rp1) ，其中包含了三个文件夹：

    - train：用于训练脚本的文件，5604张图像，包含对应的标签图像，原图像以`xxx_sat.jpg`命名，对应的标签图像则以`xxx_mask.png`命名。
    - valid：用于测试脚本的文件，622张图像，不包含对应的标签图像，原图像以`xxx_sat.jpg`命名。
    - valid_mask：用于评估脚本的文件，622张图像，是valid中图像对应的标签图像，以`xxx_mask.png`命名。
    
## 训练
- 修改dlinknet_config.yaml文件，设置下载的Resnet34/50预训练权重路径

  ```yaml
  pretrained_ckpt: '/xxx/resnet34_xxx.ckpt'
  ```
- 训练
  ```shell
  # 单卡训练命令
  python train.py --data_dir=[DATASET] --config=[CONFIG_PATH] --output_path=[OUTPUT_PATH] > train.log 2>&1 &
   ```
  参数说明:
  
  - data_dir: 训练数据集的路径
  - config: 训练配置yaml文件路径
  - output_path: 训练输出checkpoint路径
  
  ```shell
    # 训练脚本
    bash scripts/run_standalone_ascend_train.sh [DATASET] [CONFIG_PATH]
  
    # 分布式训练
    bash scripts/run_distribute_ascend_train.sh [WORKER_NUM] [DATASET] [CONFIG_PATH]
  ```
  参数说明:

  - WORKER_NUM: 用于训练的卡数
- 评估
  ```shell
    # 评估示例
    python eval.py --data_dir=[DATASET] --label_path=[LABEL_PATH] --trained_ckpt=[CHECKPOINT] --predict_path=[PREDICT_PATH] --config=[CONFIG_PATH] > eval.log 2>&1 &
  
    # 评估脚本启动
    bash scripts/run_standalone_ascend_eval.sh [DATASET] [LABEL_PATH] [CHECKPOINT] [PREDICT_PATH] [CONFIG_PATH]
  ```
    参数说明:

  - data_dir: 验证集原始图像路径
  - label_path: 验证集标签路径
  - trained_ckpt: 训练后的checkpoint路径
  - predict_path: 评估预测的结果存放路径
  - config: 训练配置路径

## Performance
- 在Ascend 910*上使用mindspore 2.4图模式测试的实验数据。

  | model name | backbone | cards | batch size | resolution | graph compile | jit level | s/step | img/s | IoU | yaml | weight |
  |:----------:|:--------:|:-----:|:----------:|:----------:|:-------------:|:---------:|:------:|:-----:|:---:|:----:|:------:|
  | dlinknet34 | resent34 | 1 | 4 | 1024x1024 |  56s | O0 | 0.16 | 25.00 | 98.19% |[yaml](./configs/dlinknet34_config.yaml)| [weight](https://download-mindspore.osinfra.cn/toolkits/models/dlinknet/dlinknet34_ascend_v4_ms2.4_resnet34_bs4_iou98.19.ckpt) |
  | dlinknet50 | resent50 | 1 | 4 | 1024x1024 | 133s | O0 | 0.38 | 10.52 | 98.18% |[yaml](./configs/dlinknet50_config.yaml)| [weight](https://download-mindspore.osinfra.cn/toolkits/models/dlinknet/dlinknet50_ascend_v4_ms2.4_resnet50_bs4_iou98.18.ckpt) |

## 评估结果样例

| 原始图像 | 标签图像 | 模型预测结果 |
|:--------------:|:-----:|:-----------------:|
|![dlinknet_999667_sat](https://github.com/user-attachments/assets/31b9e722-c44d-47bd-9c65-321420a2c4da)|![dlinknet_999667_mask](https://github.com/user-attachments/assets/355c4b81-5939-4cf4-ada6-ba45c8accc88)|![dlinknet_999667_predict](https://github.com/user-attachments/assets/57b7a05b-8aa8-41a8-a0f3-9843e19556da)|
|![dlinknet_999764_sat](https://github.com/user-attachments/assets/2f86ef1d-068a-4fb4-b9fa-33d79af51f0c)|![dlinknet_999764_mask](https://github.com/user-attachments/assets/8c9fa21d-e3d8-4b3d-9b9c-5e329bd1c0fb)|![dlinknet_999764_predict](https://github.com/user-attachments/assets/176a2d95-8fb3-441b-9d20-b42f0472ecb8)|

