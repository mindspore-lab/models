# D-LinkNet

## D-Linknet Description

D-Linknet model is constructed based on LinkNet architecture. This implementation is as described  in the original paper [D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.html).
The model performed best in the 2018 DeepGlobe Road Extraction Challenge. The network uses encoder-decoder structure, cavity convolution and pre-trained encoder to extract road.

## Requirements
 | mindspore | ascend driver | firmware | cann toolkit/kernel |
 |:---------:|:-------------:|:--------:|:-------------------:|
 | 2.3.1 | 24.1.rc2 | 7.3.0.2.220 | 8.0.RC2.beta1 |
 | 2.4.0 | 24.1.rc3 | 7.5.0.1.129 | 8.0.RC3.beta1 |
 ```shell
 pip install -r requirement.txt
 ```

## Pretrained Weights
- [Resnet34 Imagenet2012 Checkpoint](https://download-mindspore.osinfra.cn/toolkits/mindcv/resnet/resnet34-f297d27e.ckpt)
- [Resnet50 Imagenet2012 Checkpoint](https://download-mindspore.osinfra.cn/toolkits/mindcv/resnet/resnet50-f369a08d-910v2.ckpt)

## Dataset

Dataset used： [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/balraj98/deepglobe-road-extraction-dataset)

- Description: The dataset consisted of 6226 training images, 1243 validation images and 1101 test images. The resolution of each image is 1024×1024. The dataset is represented as a dichotomous segmentation problem, where roads are marked as foreground and other objects as background.
- Dataset size: 3.83 GB

    - Train: 2.79 GB, 6226 images, including the corresponding label image, the original image named 'xxx_sat.jpg', the corresponding label image named 'xxx_mask.png'.
    - Val: 552 MB, 1243 images, no corresponding label image, original image named 'xxx_sat.jpg'.
    - Test: 511 MB, 1101 images, no corresponding label image, original image named 'xxx_sat.jpg'.

- Note: since this data set is used for competition, the label images of the verification set and test set will not be disclosed. I have adopted the method of dividing the training set by one tenth as the verification set to verify the training accuracy of the model.
- The data set shown above is linked to the Kaggle community and can be downloaded directly.

- If you don't want to divide the training set by yourself, you can just download this [baiduNetDisk link](https://pan.baidu.com/s/1DofqL6P13PEDGUvNMPo-1Q?pwd=5rp1) , which contains three folders:

    - train: file used for training script, 5604 images, including the corresponding label image, the original image is named `xxx_sat.jpg`, the corresponding label image is named `xxx_mask.png`.
    - valid: file used for the test script. 622 images, not containing the corresponding label image. The original image is named `xxx_sat.jpg`.
    - valid_mask: file used for the eval script. 622 images are the label image corresponding to the valid image named `xxx_mask.png`.

## Training
- Modify the dlinknet_config.yaml file and set the download Resnet34 pretraining weight path

  ```yaml
  pretrained_ckpt: '/xxx/resnet34_xxx.ckpt'
  ```
- Training
  ```shell
  # standalone training command
  python train.py --data_dir=[DATASET] --config=[CONFIG_PATH] --output_path=[OUTPUT_PATH]  > train.log 2>&1 &
   ```
  Parameter Description:

  - data_dir: The training image folder path
  - config: The training configuration YAML file path
  - output_path: The trained output checkpoint file path
  
  ```shell
    # standalone training script
    bash scripts/run_standalone_ascend_train.sh [DATASET] [CONFIG_PATH]

    # distributed training script
    bash scripts/run_distribute_ascend_train.sh [WORKER_NUM] [DATASET] [CONFIG_PATH]
  ```
  Parameter Description:

  - WORKER_NUM: The number of cards used for training
- Evaluate
  ```shell
    # evaluation command
    python eval.py --data_dir=[DATASET] --label_path=[LABEL_PATH] --trained_ckpt=[CHECKPOINT] --predict_path=[PREDICT_PATH] --config=[CONFIG_PATH] > eval.log 2>&1 &

    # evaluation script
    bash scripts/run_standalone_ascend_eval.sh [DATASET] [LABEL_PATH] [CHECKPOINT] [PREDICT_PATH] [CONFIG_PATH]
  ```
  Parameter Description:

  - data_dir: The image path of the valid dataset
  - label_path: The label path of the valid dataset
  - trained_ckpt: The trained checkpoint file path
  - predict_path: The storage path for predicted results
  - config: The training configuration path

## Performance
- Experiments are tested on Ascend 910* with mindspore 2.4 graph mode.

  | model name | backbone | cards | batch size | resolution | graph compile | jit level | s/step | img/s | IoU | yaml | weight |
  |:----------:|:--------:|:-----:|:----------:|:----------:|:-------------:|:---------:|:------:|:-----:|:---:|:----:|:------:|
  | dlinknet34 | resent34 | 1 | 4 | 1024x1024 |  56s | O0 | 0.16 | 25.00 | 98.19% |[yaml](./configs/dlinknet34_config.yaml)| [weight](https://download-mindspore.osinfra.cn/toolkits/models/dlinknet/dlinknet34_ascend_v4_ms2.4_resnet34_bs4_iou98.19.ckpt) |
  | dlinknet50 | resent50 | 1 | 4 | 1024x1024 | 133s | O0 | 0.38 | 10.52 | 98.18% |[yaml](./configs/dlinknet50_config.yaml)| [weight](https://download-mindspore.osinfra.cn/toolkits/models/dlinknet/dlinknet50_ascend_v4_ms2.4_resnet50_bs4_iou98.18.ckpt) |

## Some Examples Of Predicted Results

| Original Image | Label Image | Predicted Results |
|:--------------:|:-----:|:-----------------:|
|![dlinknet_999667_sat](https://github.com/user-attachments/assets/31b9e722-c44d-47bd-9c65-321420a2c4da)|![dlinknet_999667_mask](https://github.com/user-attachments/assets/355c4b81-5939-4cf4-ada6-ba45c8accc88)|![dlinknet_999667_predict](https://github.com/user-attachments/assets/57b7a05b-8aa8-41a8-a0f3-9843e19556da)|
|![dlinknet_999764_sat](https://github.com/user-attachments/assets/2f86ef1d-068a-4fb4-b9fa-33d79af51f0c)|![dlinknet_999764_mask](https://github.com/user-attachments/assets/8c9fa21d-e3d8-4b3d-9b9c-5e329bd1c0fb)|![dlinknet_999764_predict](https://github.com/user-attachments/assets/176a2d95-8fb3-441b-9d20-b42f0472ecb8)|
