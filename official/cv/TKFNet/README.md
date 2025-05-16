# TKFNet

## TKFNet Description

TKFNet model is constructed based on Resnet18 architecture. 
This implementation is as described  in the original paper 
[TKFNET: Learning Texture Key Factor Driven Feature for Facial Expression Recognition](http://arxiv.org/abs/2505.09967).

## Requirements
 ```shell
 pip install -r requirement.txt
 ```

## Pretrained Weights
- [Resnet18 Checkpoint](https://drive.google.com/file/d/1CX9jyY8lsQM0agqpZP9hyOomHsYrKCwI/view?usp=drive_link)
- Download the checkpoint and move the file to ./ckpt

## Dataset

Dataset used： 
- Download RAF-DB dataset and extract the raf-basic dir to ./datasets.
- Move list_patition_label.txt into raf-basic.
- Description: The dataset is a comprehensive dataset containing over 30,000 facial images, each annotated with one of seven basic emotions. Emphasizing spontaneous expressions captured in real-life situations, it serves as a valuable resource for emotion recognition across diverse environments

- Download KDEF dataset and extract the KDEF dir to ./datasets
- Description: The dataset  features high-quality images of 70 individuals, each portraying seven distinct emotions under controlled conditions. Its consistency and clarity make it a popular choice in facial expression and psychological research.

## Model
The complete model is shown in model.py

## Training
- Modify the src/default_config.yaml file

- Training
  ```shell
  # standalone training command
  python train.py 
   ```
