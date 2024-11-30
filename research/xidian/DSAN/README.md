# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [DSAN Description](#DSAN-description)
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
            - [Train on OFFICE-31](#train-on-OFFICE-31)
        - [Evaluation Performance](#evaluation-performance)
            - [Evaluation on OFFICE-31](#evaluation-on-OFFICE-31)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DSAN Description](#contents)

DSAN (Deep Subdomain Adaptation Network) is a deep learning method designed for target tasks where labeled data is unavailable, aiming to address the issue of global domain shift encountered in traditional deep domain adaptation methods. Traditional deep domain adaptation methods primarily focus on aligning the global distributions between source and target domains, neglecting the relationships within subclasses across different domains, leading to unsatisfactory transfer learning performance due to the loss of fine-grained information. DSAN, however, emphasizes subdomain adaptation by precisely aligning the distributions of relevant subdomains. This method learns the transfer network by aligning the distributions of related subdomains of domain-specific layer activations across different domains based on Local Maximum Mean Discrepancy (LMMD).

[Paper](https://ieeexplore.ieee.org/abstract/document/9085896)：Zhu Y, Zhuang F, Wang J, et al. Deep subdomain adaptation network for image classification[J]. IEEE transactions on neural networks and learning systems, 2020, 32(4): 1713-1722.

# [Model Architecture](#contents)

DSAN Model Architecture: The core of the DSAN model architecture lies in its unique subdomain alignment mechanism, which does not require adversarial training and converges quickly. Specifically, the DSAN model can be divided into a feature extraction layer, a subdomain alignment layer, and a classification layer. In the feature extraction layer, DSAN utilizes deep neural networks to extract feature representations from both source and target domains. Next, in the subdomain alignment layer, DSAN aligns the distributions of relevant subdomains of domain-specific layer activations across different domains by computing the Local Maximum Mean Discrepancy (LMMD) loss. The design of the LMMD loss enables DSAN to precisely capture and align the distributions of relevant subdomains, thereby enhancing the performance of transfer learning. Finally, in the classification layer, DSAN uses the aligned feature representations to classify the data from the target domain.

# [Dataset](#contents)

Dataset used:[Office]https://faculty.cc.gatech.edu/~judy/domainadapt/)
Please download and unzip the dataset and place it in the '. data/[Office]https://faculty.cc.gatech.edu/~judy/domainadapt/)' folder. After successful extraction, the '. data//OFFICE31' folder should contain the following folder: amazon, dslr, and webcam, each representing a different domain.

- amazon dataset size： 2,817 224×224 colour images in 31 classes
- dslr dataset size： 498 224×224 colour images in 31 classes
- webcam dataset size： 795 224×224 colour images in 31 classes
- Data format：Image file
- Note: Data will be processed in data_loader.py

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
Before training and evaluation, you can set the default_config_cpu.yaml. Specify the source domain dataset and target domain data in the yaml file, that is, set "src_dataset: XX" and "tgt_dataset: XX", where "XX" can be selected as either 'MNIST' or 'USPU'.

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
      # (4) Set the code directory to "/path/DSAN" on the website UI interface.
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
      # (5) Set the code directory to "/path/DSAN" on the website UI interface.
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
      # (4) Set the code directory to "/path/DSAN" on the website UI interface.
      # (5) Set the startup file to "export.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

   ```text
├── DSAN  
    |—— checkpoint                  # The folder where the weight file is located     
    |—— data         
        |——OFFICE31                 # OFFICE31 Datasets
    |── model_utils
        |——config.py                # Processing configuration parameters
        |——device_adapter.py        # Get cloud ID
        |——local_adapter.py         # Get local ID
        |——moxing_adapter.py        # Parameter processing
    |── models
        |──LoadPretrainedModel      # pretrain weights
        |──DSAN.py                  # DSAN model
        |──RESNET.py                # resnet model structure
    ├── scripts
        ├──run_eval.sh              # Shell script for evaluation on Ascend
        ├──run_eval_cpu.sh          # Shell script for evaluation on CPU
        ├──run_eval_gpu.sh          # Shell script for evaluation on GPU
        ├──run_train.sh             # Shell script for distributed training on Ascend
        ├──run_train_cpu.sh         # Shell script for training on CPU
        ├──run_train_gpu.sh         # Shell script for distributed training on GPU
    ├── default_config.yaml         # Parameter configuration
    ├── eval.py                     # Evaluation script
    ├── export.py                   # Export checkpoint file to air/Mindir
    ├── README.md                   # English descriptions about DSAN
    ├── README_CN.md                # Chinese descriptions about DSAN
    ├── requirements.txt            # Required Package
    ├── train.py                    # Training script
   ```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for DSAN

  ```python  
  device_target:'Ascend'                                   # Device running the program
  nepoch: 200                                              # Number of training epochs  
  lr: [0.001, 0.01, 0.01]                                  # Learning rate
  seed: 2021                                               # Random seeds
  weight: 0.5                                              # lmmd loss weight
  momentum: 0.9                                            # Momentum
  decay: 5e-4                                              # Decay rate
  bottleneck: True                                         # Whether to add a bottleneck layer
  log_interval: 10                                         # Record interval

  # params for dataset
  nclass: 31                                               # Number of classes
  batch_size: 32                                           # Training batch size  
  src: 'amazon'                                            # Source domain dataset 
  tar: 'webcam'                                            # Target domain dataset 
  model_root: "checkpoint"                                 # Weight file storage folder  
  dataset_path: 'data/OFFICE31'                            # Dataset storage folder
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
  Epoch:1,Step:7,Loss_lmmd:3.3463
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
  Avg Accuracy = 90.078704%
  ```

## [Export Process](#contents)

### [Export](#content)

  ```shell
  python export.py
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

#### Train on OFFICE-31

|         Parameters         |                                            Ascend                                            |
|:--------------------------:|:--------------------------------------------------------------------------------------------:|
|       Model Version        |                                             DSAN                                             |
|          Resource          |                              Ascend 910；CPU 24cores；Memory 96G;                              |
|       uploaded Date        |                                 11/06/2024 (month/day/year)                                  |
|     MindSpore Version      |                                            2.2.0                                             |
|          Dataset           |                           source domain：MNIST,target domain：USPS;                            |
|    Training Parameters     | nepoch=200,lr=[0.001, 0.01, 0.01],weight=0.5,momentum=0.9,decay=5e-4,batch_size=32; |
|         Optimizer          |                                             SGD                                             |
|       Loss Function        |                                    lmmd, Cross Entropy                                     |
|          outputs           |                                         probability                                          |
|           Speed            |                   1pc: 1631 ms/step;                   |
|         Total time         |                                       1pc: 80.48 mins                                        |
|       Parameters (M)       |                                             24.09                                             |
| Checkpoint for Fine tuning |                                      91.9M (.ckpt file)                                      |
|    Model for inference     |                            91.9M (.onnx file),  91.9M(.air file)                             |

### Evaluation Performance

#### Evaluation on OFFICE-31

| Parameters |              Ascend               |
|:-------:|:---------------------------------:|
| Model Version |               DSAN                |
| Resource  | Ascend 910；CPU 24cores；Memory 96G; |
| uploaded Date |    11/06/2024 (month/day/year)    |
| MindSpore Version |               2.2.0               |
| Dataset |         webcam, 795 images          |
| batch_size |                32                |
| outputs |            probability            |
| Accuracy |      1pc: 94.70%       |
| Model for inference |        91.9M (.onnx file)         |

# [Description of Random Situation](#contents)

In train.py, we set the random number seed using np.random.seed (SEED) in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
