# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [DANN Description](#dann-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Export Process](#export-process)
        - [Export](#export)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
            - [ADDA train on MNIST and USPS](#adda-train-on-MNIST-and-USPS)
            - [ADDA train on USPS](#adda-train-on-USPS)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DANN Description](#contents)

DANN is a classic domain adaptive network framework. The author of the paper, Yaroslav Ganin, and others first introduced the idea of adversarial into the field of Transfer learning, which is simple but effective. The author conducted experimental tests on multiple datasets and achieved state of the art results.

[Paper](https://arxiv.org/abs/1409.7495)：Ganin Y , Lempitsky V . Unsupervised Domain Adaptation by Backpropagation[C]// JMLR.org. JMLR.org, 2014.

# [Model Architecture](#contents)

The backbone network of DANN is similar to LeNet, which contains a convolution layer and a maximum pooling layer. The step of the convolution layer is 1, and the number of characteristic channels is constantly increasing; The maximum pooling layer is used for downsampling. Batch normalization is used after each convolution layer, and ReLU Activation function is used after each pooling layer. Finally, output a 50 dimensional descriptor.
The class classifier of DANN is similar to the domain classifier in structure. Both of them use batch normalization and ReLU Activation function after the full connection layer. The Dropout layer needs to be used for the first full connection layer to prevent the network from over fitting.

# [Dataset](#contents)

Dataset used: [MNIST](<http://yann.lecun.com/exdb/mnist/>)
[MNIST_M](<http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz>)

- Dataset size： 70,000 28×28 gray images in 10 classes
    - Train：60,000 images  
    - Test： 10,000 images
- Data format：binary files
    - Note：Data will be processed in datasets/mnist.py

Dataset used: [USPS](https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl)  
  
- MNIST dataset introduction: this data set is a Cursive digital picture with white characters on a black background. Each picture in the data set is a gray handwritten digital picture with 28 * 28 pixels and 0~9 pixels. The image pixel value is 0~255; Images are three-dimensional data stored in bytes.
    -Training and testing: The training set contains a total of 60000 images and labels, while the testing set contains a total of 10000 images and labels
- MNIST_M dataset introduction: This dataset is a mixture of MNIST numbers and random color blocks from the BSDS500 dataset
    - Note: Data will be processed in src/data_loader.py

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorials/experts/en/master/others/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

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
  # Run training example
  python train.py > train.log 2>&1 & 

  # Run distributed training example
  bash run_train.sh [RANK_TABLE_FILE]
  # example: bash run_train.sh ~/hccl_8p.json
  
  #Run evaluation example
  python eval.py > eval.log 2>&1 &  
  OR  
  bash run_eval.sh  
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

- running on GPU

  For running on GPU, please change `device_target` from `Ascend` to `GPU` in configuration file config.yaml
  ```python
  # run training example
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &

  # run distributed training example
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7

  # run evaluation example
  python eval.py > eval.log 2>&1 &  
  OR
  bash run_eval_gpu.sh
  ```

- running on CPU

  For running on CPU, please change `device_target` from `Ascend` to `CPU` in configuration file config.yaml
  ```python
  # run training example
  bash run_train_cpu.sh
  OR
  python train.py > train.log 2>&1 &
  
  # run evaluation example
  bash run_eval_cpu.sh
  OR
  python eval.py > eval.log 2>&1 &
  ```
By default, use the MNIST dataset as the source domain, MNIST_ M dataset as the target domain. For more details, please refer to the specified script.

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    - Train 8p on ModelArts
  
      ```python
      # (1) Upload a zip dataset to S3 bucket. 
      # (2) Set the code directory to "/DANN" on the website UI interface.
      # (3) Set the startup file to "train.py" on the website UI interface.
      # (4) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (5) Create your job.
      ```

    - Eval on ModelArts

    ```python
      # (1) Upload or copy your pretrained model to S3 bucket.
      # (2) Upload a zip dataset to S3 bucket.
      # (3) Set the code directory to "/path/adda" on the website UI interface.
      # (4) Set the startup file to "eval.py" on the website UI interface.
      # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (6) Create your job.
    ```


# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
├── model_zoo
    ├── DANN
        ├── README_CN.md                // DANN related Chinese instructions
        ├── README.md                   // DANN related English instructions
        ├── model_utils
            ├── config.py               // Configuration file
        ├── models
        ├── scripts
        │   ├──run_train.sh             // Training shell script for Ascend processor
        │   ├──run_train_gpu.sh         // Training shell script for GPU processor
        │   ├──run_train_cpu.sh         // Training shell script for CPU processor
        │   ├──run_eval.sh              // Evaluation shell script for Ascend processor
        │   ├──run_eval_gpu.sh          // Evaluation shell script for GPU processor
        │   ├──run_eval_cpu.sh          // Evaluation shell script for CPU processor
        ├── src
        │   ├──dataloader.py            // Read Dataset
        │   ├──model.py                 // DANN Network Architecture
        │   ├──train_cell.py            // DANN training module
        ├── train.py                    // Training script
        ├── eval.py                     // Evaluation script
        ├── export.py                   // Export checkpoint file to air/Mindir
        ├── config.yaml                   // Configuration file
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for DANN

  ```python
  'batch_size':64                                       # Training batch size
  'lr':0.001                                            # Learning rate of classification layer in backbone pretraining, 
  'lr_backbone_s':0.002                                 # Learning rate of backbone in adversarial training
  'model_root':"./models"                               # Model Save Path
  'n_epoch':100                                         # Number of epochs for adversarial training
  'n_pretrain_epochs':100                               # Number of pretrained epochs
  'source_dataset_name':"MNIST"                         # Source Domain Dataset Name
  'source_image_root': "./dataset/MNIST"                # Source Domain Dataset Path
  'target_dataset_name':"mnist_m"                       # Target Domain Dataset Name
  'target_image_root': "./dataset/mnist_m"              # Target Domain Dataset Path
  'weight_decay':1.0e-05                                # Weight decay
  'device_target':"Ascend"                              # operating device
  'device_id':0                                         # Device ID used for training or evaluating datasets using run_train.sh can be ignored for distributed training.
  'backbone_ckpt_file':"./models/best_backbone_t.ckpt"  # The absolute path to load the checkpoint file of the backbone during inference
  'classifier_ckpt_file':"./models/best_class_classifier.ckpt"  # The absolute path of the checkpoint file for loading the classifier during inference
  'imageSize': 28                                       # Input image size
  'Linear': 800                                         # Size of fully connected layer input
  'file_name':"DANN.air"                                # Exported file name
  'file_format':"AIR"                                   # Exported file format
  ```


For more configuration details, please refer the script `config.py`.

## [Training Process](#contents)

### Training

- running on Ascend

  ```bash
  python train.py > train.log 2>&1 &
  ```

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```bash  
  # grep "loss is " train.log  
  epoch: 0, [iter: 91 / all 921],  err_D_domain: 1.405273, err_G_domain: 0.480713,err_sum:1.885742  
  ```   

- running on GPU

  For running on GPU, please change `device_target` from `Ascend` to `GPU` in configuration file config.yaml
  ```bash  
  export CUDA_VISIBLE_DEVICES=0  
  python train.py > train.log 2>&1 &  
  OR
  bash run_train_gpu.sh 8 0,1,2,3,4,5,6,7  
  ```

- running on CPU

   For running on CPU,  please change `device_target` from `Ascend` to `CPU` in configuration file config.yaml
  ```bash  
  python train.py  > train.log 2>&1 &  
  OR
  bash run_train_cpu.sh
  ```
  All the shell command above will run in the background, you can view the results through the file `train/train.log`.

### Distributed Training

- running on GPU
   
For running on GPU, please change `device_target` from `Ascend` to `GPU` in configuration file config.yaml

  ```bash
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7
  ```

  The above shell command will run distribute training in the background. You can view the results through the file `train/train.log`.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on target dataset when running on Ascend

 ```bash  
  python eval.py > eval.log 2>&1 &  
  或  
  bash run_eval.sh  
  ```

- evaluation on target dataset when running on GPU

  For running on GPU, please change `device_target` from `Ascend` to `GPU` in configuration file config.yaml

  ```bash  
  python eval.py > eval.log 2>&1 &  
  OR  
  bash run_eval_gpu.sh  
  ```
- evaluation on target dataset when running on CPU

  For running on CPU, please change `device_target` from `Ascend` to `CPU` in configuration file config.yaml
  ```bash  
  python eval.py > eval.log 2>&1 &  
  或  
  bash run_eval_cpu.sh  
  ```  
  
  The above shell command will run in the background. You can view the results through the file `eval/eval.log`. The accuracy of the test dataset will be as follows:

  ```bash  
  source domain accuracy: 0.994591
  target domain accuracy: 0.815737 
  ```  

## [Export Process](#contents)

### [Export](#content)

  ```shell  
  python export.py
  ```  

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

#### GoogleNet train on MNIST and USPS

|         Parameters         |                                       Ascend                                        |
|:--------------------------:|:-----------------------------------------------------------------------------------:|
|       Model Version        |                                        DANN                                         |
|          Resource          |                         Ascend 910；CPU 24cores；Memory 96G;                          |
|       uploaded Date        |                             07/04/2023 (month/day/year)                             |
|     MindSpore Version      |                                        1.8.1                                        |
|          Dataset           |                     source domain：MNIST,target domain：MNIST_M;                      |
|    Training Parameters     |  batch_size: 64,lr: 0.001,lr_backbone_s: 0.002,n_epoch: 100,n_pretrain_epochs: 50;  |
|         Optimizer          |                                        Adam                                         |
|       Loss Function        |                                Softmax Cross Entropy                                |
|          outputs           |                                     probability                                     |
|           Speed            |             1pc: 130 ms/step(source domain),130 ms/step(target domain);             | 
|         Total time         |                                    1pc: 204 mins                                    | 
|       Parameters (M)       |                                        0.176                                        |
| Checkpoint for Fine tuning |                                 0.696M (.ckpt file)                                 | 
|    Model for inference     |                      0.714M (.mindir file),  0.732M(.air file)                      |
|          Scripts           | [DANN script](https://gitee.com/mindspore/models/tree/master/official/cv/googlenet) | [googlenet script](https://gitee.com/mindspore/models/tree/master/official/cv/googlenet) |

### Evaluation Performance

#### Evaluation on USPS

| Parameters |               Ascend               |
|:-------:|:----------------------------------:|
| Model Version |                DANN                |
| Resource  | Ascend 910；CPU 24cores；Memory 96G; |
| uploaded Date |    04/07/2023 (month/day/year)     |
| MindSpore Version |               1.8.1                |
| Dataset |        MNIST_M,9000 images         |
| batch_size |                 64                 |
| outputs |            probability             |
| Accuracy |            1pc: 81.50%             |
| Model for inference |       0.714M (.mindir file)        |


# [Description of Random Situation](#contents)

In train.py, we use init_random_seed() function in utiles.py sets a random number seed.

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/models).  
