# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [VSE++ Description](#adda-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
            - [Training](#training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation)
        - [Export Process](#export-process)
            - [Export](#export)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
            - [Train on Flickr30K](#train-on-mnist-and-usps)
        - [Evaluation Performance](#evaluation-performance)
            - [Evaluation on Flickr30K](#evaluation-on-usps)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [VSE++ Description](#contents)

The VSE++(Improving Visual-Semantic Embedding with Hard Negatives) method was put forward in 2018 to solve the problem of cross-modal text retrieval. In this paper, the author adds difficult case mining to the loss function, and uses difficult case mining to minimize the loss function, without any additional mining cost, and realizes the improvement of performance.

[Paper](https://link.zhihu.com/?target=http://www.bmva.org/bmvc/2018/contents/papers/0344.pdf)：Faghri F, Fleet D J, Kiros J R, et al. Vse++: Improving visual-semantic embeddings with hard negatives[J]. arXiv preprint arXiv:1707.05612, 2017.

# [Model Architecture](#contents)

The main goal of this model is to match the image with the text, that is, to associate the image with the corresponding text description. It adopts VGG19 as the image encoder and single-layer RGU as the text encoder. VGG19 is a classical convolutional neural network, which consists of 19 convolutional layers and 3 fully connected layers. It performs well in large-scale image classification tasks and has been widely used in various computer vision tasks. In this model, VGG19 is responsible for transforming the input image into a series of image feature vectors. Single-layer RGU is a kind of encoder based on Recurrent Neural Network. It can model the input text sequence data and output a fixed-length vector representation.

# [Dataset](#contents)

The model uses Flickr30K data set. Flickr30K is a commonly used image description data set, which contains 30,000 images from Flickr website, and each image has a corresponding text description. This data set is widely used in the task of association learning between images and texts, such as image description generation and image retrieval. Please use the following command to download the data set and unzip it and put it in the `. data/ ` folder.
```
wget http://www.cs.toronto.edu/~faghri/vsepp/data.tar
```
The `. data/ ` folder after successful decompression should contain the following folders: f30k, f30k_precomp.

-Data set size: Flickr30K data set contains about 31,000 images, and each image corresponds to 5 descriptive labels, totaling about 150,000 labels.
-data format: npy and txt files
# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend/GPU/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:
- running on Ascend

```python
#  run training example
  python train.py --device_id 0  > train_Ascend.log 2>&1 &
 OR
  bash scripts/run_train.sh
   # run evaluation example
  python eval.py --device_id 0  > eval_Ascend.log 2>&1
  OR
  bash scripts/run_eval.sh
```
- running on CPU

```python
#  run training example
  bash run_train_cpu.sh
  OR
  python train.py --device "CPU"  > train_CPU.log 2>&1 &

 # run evaluation example
  bash run_eval_cpu.sh
  OR
  python eval.py --device "CPU"  > eval_CPU.log 2>&1 &
```
# [Script Description](#contents)

## [Script and Sample Code](#contents)

   ```text
├── vsepp  
    |—— data
        |——f30k_precomp             # f30k_precomp dataset
    |—— src
        |——data.py                  # deal with f30k_precomp dataset
        |——trainer.py                # Building a trainer
|——vocab.py                 # Processing text encoding
|——f30k_precomp_vocab.pkl    # Text correspondence of data set image
    |── models
        |──model.py         		 # Main model structure
        |──model_vgg19.py           # Vgg19 model structure
        |──var_init.py                # Model weight initialization method
|── model_utils
        |——config.py                 # Processing configuration parameters
        |——device_adapter.py          # Get cloud ID
        |——local_adapter.py           # Get local ID
        |——moxing_adapter.py         # Parameter processing
    ├── scripts
        ├──run_eval.sh              # Shell script for Ascend evaluation
        ├──run_eval_cpu.sh          # CPU处理器评估的shell脚本
        ├──run_train.sh              # Shell script for CPU processor evaluation
        ├──run_train_cpu.sh          # Shell script for CPU training
├── default_config.yaml          # Parameter configuration file
    ├── eval.py                    # Evaluation script
    ├── export.py                   # Export the checkpoint file to air/mindir.
    ├── README.md                # English explanation
    ├── README_CN.md             # Chinese explanation
    ├── requirements.txt             # Required package
    ├── train.py                    # Training script

   ```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in default_config.yaml

- config for VSE++

```python
batch_size: 128
epoch: 30
learning rate: 0.0002
lr_update: 15
optimizer: Adam
margin: 0.2
word_dim: 300
embed_size: 1024
grad_clip: 2
crop_size: 224
num_layers: 1
max_violation: true
img_dim: 4096
workers: 1
log_step: 200
val_step: 500
cnn_type: "vgg19"
use_abs: false
no_imgnorm: false
seed: 123
device_id: 7
device_target: "Ascend"
data_path: "data"
data_name: "f30k_precomp"
vocab_path: "./vocab/"
num_epochs: 30
use_restval: false
enable_modelarts: false
finetune: false
ckpt_file: "./best.ckpt"
vocab_size: 8481
file_name: "model"
file_format: "MINDIR"
```


更多配置细节请参考脚本`default_config.yaml`。  

## [Training Process](#contents)

### Training

- running on Ascend

 
  ```bash
  python train.py --device_id 0  > train_Ascend.log 2>&1 &
  或
  bash scripts/run_train.sh

After the training, you can get the following results:

```python
================epoch :30================
[2024-03-09 10:35:59] step: [1/1132] loss: 12.309782 lr: 0.00002
[2024-03-09 10:36:19] step: [201/1132] loss: 9.338764 lr: 0.00002
[2024-03-09 10:36:38] step: [401/1132] loss: 9.604823 lr: 0.00002
[2024-03-09 10:36:58] step: [601/1132] loss: 9.094549 lr: 0.00002
[2024-03-09 10:37:17] step: [801/1132] loss: 9.84281 lr: 0.00002
[2024-03-09 10:37:36] step: [1001/1132] loss: 10.121641 lr: 0.00002
Per step costs time(ms): 0:00:00.096494
0/39
10/39
20/39
30/39
Image to text: 26.7, 52.4, 61.6, 5.0, 42.1
Text to image: 20.2, 42.3, 52.6, 9.0, 63.6
rsum:  255.8517034068136
Best score: 299.47895791583164
```
## [Evaluation Process](#contents)

### Evaluation

- evaluation on target dataset when running on Ascend

```python
  python eval.py --device_id 0  > eval_Ascend.log 2>&1
  或
  bash scripts/run_eval.sh
```
All the above commands will run in the background, and you can view the results through the log file. The accuracy of the test data set is as follows:
```python
rsum: 295.3
Average i2t Recall: 53.6
Image to text: 31.7 57.9 71.1 4.0 26.9
Average t2i Recall: 44.9
Text to image: 23.4 49.4 61.8 6.0 34.8
```
## [Export Process](#contents)

### [Export](#content)

  ```shell
  python export.py
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

#### Train on MNIST and USPS

|         Parameters         |                                            Ascend                                            |
|:--------------------------:|:--------------------------------------------------------------------------------------------:|
|       Model Version        |                                            VSE++                                           |
|          Resource          |                               Ascend 910；CPU 2.60GHz，72核；内存 503G;                                  |                            |
|       uploaded Date        |                                   2023-03-18                                         |
|     MindSpore Version      |                                            1.10.1                                             |
|          Dataset           |                            Flickr30K                         |
|    Training Parameters     | batch_size=128；epoch=30；learning rate=0.0002；loss function= ContrastiveLoss；optimizer=Adam；margin=0.2；word_dim=300；embed_size=1024；grad_clip=2；crop_size=224；num_layers=1；max_violation=true；img_dim=40964; |
|         Optimizer          |                                             Adam                                             |
|       Loss Function        |                                   Max Hinge Loss                                    |
|          outputs           |                                         Recall rate                                          |
|           Speed            |                   95ms/step(1p)                   |
|         Total time         |                                        48.5 mins(1p)                                        |              |
|    Model for inference     |                            41.78M(.air文件)， 41.76(.mindir文件）                      |

# [Description of Random Situation](#contents)

In train.py, we use init_random_seed() function in utiles.py sets a random number seed.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
