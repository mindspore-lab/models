# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [FADA Description](#fada-description)
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
            - [Train on MNIST and SVHN](#train-on-mnist-and-svhn)
        - [Evaluation Performance](#evaluation-performance)
            - [Evaluation on SVHN](#evaluation-on-svhn)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [FADA Description](#contents)

The FADA method is a deep learning method proposed in 2017 for domain adaptation. The application scenario of this paper is that only a small amount of annotated target domain data is available. To address the lack of training samples in the target domain, the author constructed pairs of these samples along with each source domain training sample, which can be divided into four groups:
- 1、The samples that make up a pair are all from the source domain and have the same labels;
- 2、The paired samples come from both the source and target domains, with the same labels;
- 3、The samples that make up a pair all come from the source domain and have different labels;
- 4、The samples that make up a pair come from both the source and target domains, with different labels;

[Paper](https://papers.nips.cc/paper_files/paper/2017/hash/21c5bba1dd6aed9ab48c2b34c1a0adde-Abstract.html)：Motiian S, Jones Q, Iranmanesh S, et al. Few-shot adversarial domain adaptation[J]. Advances in neural information processing systems, 2017, 30.

# [Model Architecture](#contents)

FADA is divided into three steps. The first step is to pretrain with the source domain, initialize g and h, and minimize classification loss. The second step is to freeze g and train a domain class discriminator (DCD). The four classifications mentioned earlier confuse the classification of these four sample pairs to the greatest extent possible. The third step is to fix the DCD and update g and h. Finally, the FADA method uses the trained g and g to classify the data in the target domain.

# [Dataset](#contents)

Dataset used: [MNIST](http://yann.lecun.com/exdb/mnist/)
Please download and unzip the dataset and place it in the `.dataset/MNIST_Data` folder. After successful extraction, the `.data/MNIST_Data` folder should contain the following files: t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte, train-images-idx3-ubyte, train-labels-idx1-ubyte.

- Dataset size： 70,000 28×28 gray images in 10 classes
    - Train：60,000 images  
    - Test： 10,000 images
- Data format：binary files
    - Note：Data will be processed in model/dataloader.py

Dataset used: [SVHN](http://ufldl.stanford.edu/housenumbers/)  
Please download the dataset and place it in the `.dataset/SVHN` folder. The successfully downloaded `.data/SVHN` folder should contain the following files: test_32x32.mat,train_32x32.mat.

- Dataset size: 99289 28×28 gray images in 10 classes
    - Train：73,257 images  
    - Test： 26,032 images
- Data format：mat
    - Note: Data will be processed in model/dataloader.py

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
Before training and evaluation, you can set the default_config_cpu.yaml. Specify the source domain dataset and target domain data in the yaml file, that is, set "src_dataset: XX" and "tgt_dataset: XX", where "XX" can be selected as either 'MNIST' or 'SVHN'.

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
      # (4) Set the code directory to "/path/fada" on the website UI interface.
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
      # (5) Set the code directory to "/path/fada" on the website UI interface.
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
      # (4) Set the code directory to "/path/fada" on the website UI interface.
      # (5) Set the startup file to "export.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

  ```text
├── FADA  
  ├── checkpoint
  ├── core
  │   ├── step1.py              # 第一步训练
  │   ├── step2.py              # 第二步训练
  │   └── step3.py              # 第三步训练
  ├── dataset
  │   ├── MNIST_Data            # MNIST数据集
  │   └── SVHN                  # SVHN数据集
  ├── model
  │   ├── dataloader.py         # 处理数据集，构建Dataloader
  │   ├── model.py              # 模型结构
  │   └── utils.py              # 工具文件
  ├── model_utils
  │   ├── config.py             # 处理配置参数
  │   ├── device_adapter.py     # 获取云ID
  │   ├── local_adapter.py      # 获取本地ID
  │   ├── moxing_adapter.py     # 参数处理
  │   └── utils.py              # 工具文件
  ├── scripts
  │   ├── run_eval_cpu.sh       # CPU处理器评估的shell脚本
  │   ├── run_eval_gpu.sh       # GPU处理器评估的shell脚本
  │   ├── run_eval.sh           # Ascend评估的shell脚本
  │   ├── run_train_cpu.sh      # 用于CPU训练的shell脚本
  │   ├── run_train_gpu.sh      # 用于GPU上运行分布式训练的shell脚本
  │   └── run_train.sh          # 用于分布式训练的shell脚本 
  ├── README_CN.md              # FADA相关中文说明
  ├── README.md                 # FADA相关英文说明
  ├── requirements.txt          # 需要的包
  ├── default_config.yaml       # 参数配置文件
  ├── eval.py                   # 评估脚本
  ├── export.py                 # 将checkpoint文件导出到air/mindir
  └── train.py                  # 训练脚本
  ```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for FADA

  ```python
  enable_modelarts: False 
  device_target: "Ascend"                                         # Device running the program  
  model_root: "checkpoint"
  src_encoder_checkpoint: "FADA-source-encoder.ckpt"              # Source domain encoder weight file
  src_classifier_checkpoint: "FADA-source-classifier.ckpt"        # Source domain classifier weight file  
  tgt_discriminator_checkpoint: "FADA-tgt-discriminator.ckpt"     # Target domain discriminator weight file
  tgt_encoder_checkpoint: "FADA-tgt-encoder.ckpt"                 # Target domain encoder weight file 
  tgt_classifier_checkpoint: "FADA-tgt-classifier.ckpt"           # Target domain classifier weight file 
  n_epoch_1: 10                                                   # Number of epochs for step1
  n_epoch_2: 100                                                  # Number of epochs for step2 
  n_epoch_3: 100                                                  # Number of epochs for step3
  disc_feature: 128                                               # Discriminator Hidden Layer Dimension
  src_lr: 1e-3                                                    # Discriminator output Layer Dimension
  dcd_lr_2: 1e-3                                                  # Discriminator learning rate of step 2
  CE_lr_3: 3e-3                                                   # Encoder and Classifier learning rate of step 3
  D_lr_3: 3e-3                                                    # Discriminator learning rate of step 3 
  loss_dcd_weight: 0.2                                            # Weight of discriminator loss
  n_target_samples: 7                                             # numbers of paired samples
  batch_size: 64                                                  # Training batch size
  mini_batch_size_g_h: 20                                         # Minimum batch size for encoders and classifiers
  mini_batch_size_dcd: 40                                         # The minimum batch size of discriminator
  file_name: "net"                                                # Export file name  
  file_format: "MINDIR"                                           # Export file format  
  image_height: 28                                                # The height of the sample data image 
  image_width: 28                                                 # The width of the sample data image
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
  Epoch [1/10] Step [600]: loss=0.14988492  Epoch [1/10] Step [1200]: loss=0.124895595
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
  Avg Accuracy = 47.027027%
  ```

## [Export Process](#contents)

### [Export](#content)

  ```shell
  python export.py
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

#### Train on MNIST and SVHN

|         Parameters         |                                            Ascend                                            |
|:--------------------------:|:--------------------------------------------------------------------------------------------:|
|       Model Version        |                                             FADA                                             |
|          Resource          |                              Ascend 910；CPU 24cores；Memory 96G;                              |
|       uploaded Date        |                                 02/01/2024 (month/day/year)                                  |
|     MindSpore Version      |                                            1.10.1                                             |
|          Dataset           |                           source domain：MNIST,target domain：SVHN;                            |
|    Training Parameters     |  n_epoch_1=10,n_epoch_2=60,n_epoch_3=60,batch_size=64,src_lr=1e-3,dcd_lr_2=1.0e-3; |
|         Optimizer          |                                             Adam                                             |
|       Loss Function        |                                    Softmax Cross Entropy                                     |
|          outputs           |                                         probability                                          |
|           Speed            |                   1pc: 6.176 ms/step(step 1),0.142 ms/step(step 2),446.702 ms/step(step 3);                   |
|         Total time         |                                       1pc: 182.45 mins                                        |
|       Parameters (M)       |                                             49.444k                                             |
| Checkpoint for Fine tuning |                                      0.19M (.ckpt file)                                      |
|    Model for inference     |                            0.19M (.onnx file),  0.21M(.air file)                             |

### Evaluation Performance

#### Evaluation on SVHN

| Parameters |              Ascend               |
|:-------:|:---------------------------------:|
| Model Version |               FADA                |
| Resource  | Ascend 910；CPU 24cores；Memory 96G; |
| uploaded Date |    02/01/2024 (month/day/year)    |
| MindSpore Version |               1.10.1               |
| Dataset |         SVHN,26032 images          |
| batch_size |                64                |
| outputs |            probability            |
| Accuracy |      1pc: 47.15%       |
| Model for inference |        0.19M (.onnx file)         |

# [Description of Random Situation](#contents)

In train.py, we use init_random_seed() function in utiles.py sets a random number seed.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
