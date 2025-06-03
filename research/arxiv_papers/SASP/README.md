# SASP

## SASP Description

SASP model is constructed based on Resnet50 architecture. 

## Requirements
 ```shell
 pip install -r requirements.txt
 ```

## Dataset

Dataset used：
- Download CUB-200-2011 dataset and extract the CUB-200-2011 dir to ./datasets
- Description: The CUB-200-2011 dataset comprises 200 bird species with a total of 11,788 images, of which 5,994 are allocated for training and 5,794 for testing.

## Model
The complete model is shown in model.py

## Training
- Modify the src/default_config1.yaml file

- Training
  ```shell
  # standalone training command
  python main.py --cub_root your_file_path/CUB200-2011
   ```
