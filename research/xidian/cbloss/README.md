# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [CBLOSS Description](#aCBLOSS-description)
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
            - [Train on Cifar-10](#train-on-Cifar-10)
        - [Evaluation Performance](#evaluation-performance)
            - [Evaluation on Cifar-10](#evaluation-on-Cifar-10)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CBLOSS Description](#contents)

The CBLOSS method is a deep learning optimization strategy tailored for the problem of long-tailed data distributions, proposed in 2019, which aims to address the decline in model performance caused by uneven category distributions in datasets. In long-tailed distributions, a few categories dominate, while most categories are underrepresented, significantly affecting the model's recognition ability for minority categories. The CBLOSS method introduces an innovative class-balanced loss function that redefines the importance of each sample, thereby enhancing the overall performance of the model on long-tailed datasets. The core of this method lies in using the concept of effective sample size to weight each sample, compensating for the training bias caused by differences in category sample numbers.

[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)：Cui Y, Jia M, Lin T Y, et al. Class-balanced loss based on effective number of samples[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 9268-9277.

# [Model Architecture](#contents)

The core of the CBLOSS method lies in its unique loss function design, rather than a specific model architecture. It is applicable to various deep learning models and can be integrated as the training loss function for these models. Specifically, the implementation process of CBLOSS can be divided into the following steps: Firstly, for a given long-tailed dataset, calculate the effective sample size for each category, which is derived using the formula (1-b^n) / (1-b), where n is the number of samples and b is an adjustable hyperparameter that controls the influence of sample overlap. Next, based on the effective sample sizes, design a reweighting scheme that assigns different weights to samples of different categories to balance their contributions during training. Finally, apply this reweighted loss function to the training of deep learning models, prompting the models to learn more balanced feature representations and classification abilities on long-tailed datasets.

# [Dataset](#contents)

Dataset used: [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html)
Please download and unzip the dataset and place it in the '. data/cifar10_dataset_directory' folder. After successful extraction, the '. data/cifar10_dataset_directory' folder should contain the following files: batches.meta.txt, data_batch_1.bin, data_batch_2.bin, data_batch_3.bin,data_batch_4, bindata_batch_5.bin, test_batch.bin.

- Dataset size： 60,000 32×32 colour images in 10 classes
    - Train：50,000 images  
    - Test： 10,000 images
- Data format：binary files
    - Note：Data will be processed in dataset.py


# [Features](#contents)

## Mixed Precision

The mixed precision training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
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

  For running on GPU, please change `device_target` from `Ascend` to `GPU` in configuration file default_config.yaml

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

  For running on CPU, please change `device_target` from `Ascend` to `CPU` in configuration file default_config.yaml

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

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    - Train 8p on ModelArts

    ```python
      # (1) Add "config_path='./default_config.yaml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=True" on default_config.yaml file.
      #          Set other parameters on default_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add other parameters on the website UI interface.
      # (3) Upload a zip dataset to S3 bucket.
      # (4) Set the code directory to "/path/CBLOSS" on the website UI interface.
      # (5) Set the startup file to "train.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
    ```

    - Eval on ModelArts

    ```python
      # (1) Add "config_path='./default_config.yaml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=True" on default_config.yaml file.
      #          Set other parameters on default_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add other parameters on the website UI interface.
      # (3) Upload or copy your pretrained model to S3 bucket.
      # (4) Upload a zip dataset to S3 bucket.
      # (5) Set the code directory to "/path/CBLOSS" on the website UI interface.
      # (6) Set the startup file to "eval.py" on the website UI interface.
      # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (8) Create your job.
    ```

    - Export on ModelArts

    ```python
      # (1) Add "config_path='./default_config.yaml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=True" on default_config.yaml file.
      #          Set other parameters on cifar10_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add other parameters on the website UI interface.
      # (3) Upload or copy your trained model to S3 bucket.
      # (4) Set the code directory to "/path/CBLOSS" on the website UI interface.
      # (5) Set the startup file to "export.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

   ```text
├── CBLOSS  
    |—— checkpoint                  # Weight file storage folder 
    |—— data
        |——cifar10_dataset_directory# Cifar10 dataset
    |── model_utils
        |——config.py                # Processing configuration parameters
        |——device_adapter.py        # Get cloud ID
        |——local_adapter.py         # Get local ID
        |——moxing_adapter.py        # Parameter processing
    |── models
        |──Resnet.py                # model Structure
    |── pretrained_weights          # Preliminary training weights in the folder
    ├── scripts
        ├──run_eval.sh              # Shell script for evaluation on Ascend
        ├──run_eval_cpu.sh          # Shell script for evaluation on CPU
        ├──run_eval_gpu.sh          # Shell script for evaluation on GPU
        ├──run_train.sh             # Shell script for distributed training on Ascend
        ├──run_train_cpu.sh         # Shell script for training on CPU
        ├──run_train_gpu.sh         # Shell script for distributed training on GPU
    ├── CBloss.py                   # CBLOSS
    ├──dataset.py                   # deal with cifar10 datasets     
    ├── default_config.yaml         # Parameter configuration
    ├── eval.py                     # Evaluation script
    ├── export.py                   # Export checkpoint file to air/Mindir
    ├── README_CN.md                # Chinese descriptions about CBLOSS
    ├── README.md                   # English descriptions about CBLOSS
    ├── requirements.txt            # Required Package
    ├── train.py                    # Training script
   ```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for CBLOSS

  ```python  
  device_target:'Ascend'                                   # Device running the program
  model_root: "checkpoint"                                 # Weight file storage folder  
  pretrained_model: "pretrained_weights"                   # Pre-training weight file storage folder 
  checkpoint_path: "model.ckpt"                            # Weight file name 
  epoch_num: 1000                                          # Training epoch number  
  data_root: "data"                                        # Dataset storage folder
  log_step: 10                                             # Record interval number
  imb_ratio: 100                                           # Imbalance ratio
  batch_size: 50                                           # train batch size    
  loss_type: 'focal'                                       # CBLOSS type
  beta: 0.9999                                             # CBLOSSParameter
  gamma: 2                                                 # CBLOSSParameter
  LR: 0.01                                                 # Learning rate
  file_name: "net"                                         # Export file name
  file_format: "ONNX"                                      # Export file format
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
  Train Epoch: 1 [0/132 (0%)]	Loss: 0.141106
  ```

- running on GPU

  For running on GPU, please change `device_target` from `Ascend` to `GPU` in configuration file default_config.yaml

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &
  OR
  bash run_train_gpu.sh 8 0,1,2,3,4,5,6,7
  ```

- running on CPU

   For running on CPU,  please change `device_target` from `Ascend` to `CPU` in configuration file default_config.yaml

  ```bash
  python train.py  > train.log 2>&1 &
  OR
  bash run_train_cpu.sh
  ```

  All the shell command above will run in the background, you can view the results through the file `train/train.log`.

### Distributed Training

- running on GPU

For running on GPU, please change `device_target` from `Ascend` to `GPU` in configuration file default_config.yaml

  ```bash
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7
  ```

  The above shell command will run distribute training in the background. You can view the results through the file `train/train.log`.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on target dataset when running on Ascend

 ```bash  
  python eval.py > eval.log 2>&1 &
  OR
  bash run_eval.sh
  ```

- evaluation on target dataset when running on GPU

  For running on GPU, please change `device_target` from `Ascend` to `GPU` in configuration file default_config.yaml

  ```bash
  python eval.py > eval.log 2>&1 &
  OR  
  bash run_eval_gpu.sh
  ```

- evaluation on target dataset when running on CPU

  For running on CPU, please change `device_target` from `Ascend` to `CPU` in configuration file default_config.yaml

  ```bash
  python eval.py > eval.log 2>&1 &
  OR
  bash run_eval_cpu.sh
  ```  

  The above shell command will run in the background. You can view the results through the file `eval/eval.log`. The accuracy of the test dataset will be as follows:

  ```bash
  Test set: Accuracy: 89%
  ```

## [Export Process](#contents)

### [Export](#content)

  ```shell
  python export.py
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

#### Train on Cifar-10

|         Parameters         |                                            Ascend                                            |
|:--------------------------:|:--------------------------------------------------------------------------------------------:|
|       Model Version        |                                             CBLOSS                                             |
|          Resource          |                              Ascend 910；CPU 24cores；Memory 96G;                              |
|       uploaded Date        |                                 11/03/2024 (month/day/year)                                  |
|     MindSpore Version      |                                            2.2.0                                             |
|          Dataset           |                           Cifar-10;                            |
|    Training Parameters     | num_epochs = 500, batch_size = 128 LR = 0.01; |
|         Optimizer          |                                             SGD                                             |
|       Loss Function        |                                    CBLOSS                                     |
|          outputs           |                                         probability                                          |
|           Speed            |                   1pc: 42.10 ms/step;                   |
|         Total time         |                                       1pc: 53.46 mins                                        |
|       Parameters (M)       |                                             21.31                                              |
| Checkpoint for Fine tuning |                                      81.2M (.ckpt file)                                      |
|    Model for inference     |                            81.2M (.onnx file),  81.8M(.air file)                             |

### Evaluation Performance

#### Evaluation on Cifar-10

| Parameters |              Ascend               |
|:-------:|:---------------------------------:|
| Model Version |               CBLOSS                |
| Resource  | Ascend 910；CPU 24cores；Memory 96G; |
| uploaded Date |    3/11/2024 (month/day/year)    |
| MindSpore Version |               2.2.0               |
| Dataset |         Cifar-10, 10000 images          |
| batch_size |                128                |
| outputs |            probability            |
| Accuracy |      1pc: 89.22%%       |
| Model for inference |        81.2M (.onnx file)         |


# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
