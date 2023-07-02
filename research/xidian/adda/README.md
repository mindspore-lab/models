# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [ADDA Description](#adda-description)
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
            - [Train on MNIST and USPS](#train-on-mnist-and-usps)
        - [Evaluation Performance](#evaluation-performance)
            - [Evaluation on USPS](#evaluation-on-usps)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [ADDA Description](#contents)

The ADDA method is a deep learning method proposed in 2017 for domain adaptation, aimed at solving the problem of different distributions of training data and test data. In deep learning tasks, the generalization ability of the trained model on new datasets often decreases due to the distribution differences between different datasets. The ADDA method minimizes the difference between the source domain and the target domain through adversarial training, thereby improving the model's generalization ability in the target domain.

[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf)：Tzeng E, Hoffman J, Saenko K, et al. Adversarial discriminative domain adaptation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 7167-7176.

# [Model Architecture](#contents)

ADDA is divided into pre training stage and domain adaptation stage. In the pre training stage, the feature extractor and classifier in the source domain are trained using source domain data. In the domain adaptation stage, the target domain feature extractor and source domain feature extractor are used to extract features from the target domain and source domain data respectively, and adversarial training is conducted with the discriminator. The goal of the discriminator is to correctly determine whether the input features come from the source domain or target domain, while the goal of the target domain feature extractor is to confuse the discriminator, making it unable to distinguish correctly. Through this adversarial learning approach, The ADDA method can gradually reduce the distribution difference between source domain features and target domain features, thereby improving the model's generalization ability in the target domain. Finally, the ADDA method uses trained source domain feature extractors and target domain classifiers to classify data in the target domain.

# [Dataset](#contents)

Dataset used: [MNIST](http://yann.lecun.com/exdb/mnist/)
Please download and unzip the dataset and place it in the '. data/MNIST' folder. After successful extraction, the '. data/MNIST' folder should contain the following files: t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte, train-images-idx3-ubyte, train-labels-idx1-ubyte.

- Dataset size： 70,000 28×28 gray images in 10 classes
    - Train：60,000 images  
    - Test： 10,000 images
- Data format：binary files
    - Note：Data will be processed in datasets/mnist.py

Dataset used: [USPS](https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl)  
Please download the dataset and place it in the '. data/USPS' folder. The successfully downloaded'. data/USPS' folder should contain the following files: USPS_ 28x28.pkl.

- Dataset size: 9298 28×28 gray images in 10 classes
    - Train：7,438 images  
    - Test： 1,860 images
- Data format：pkl
    - Note: Data will be processed in datasets/usps.py

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
      # (4) Set the code directory to "/path/adda" on the website UI interface.
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
      # (5) Set the code directory to "/path/adda" on the website UI interface.
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
      # (4) Set the code directory to "/path/adda" on the website UI interface.
      # (5) Set the startup file to "export.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.
    ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

   ```text
├── adda  
    |—— core
        |——adapt.py                 # Train on source domain
        |——pretrain.py              # Train on target domain
    |—— data
        |——MNIST                    # MNIST dataset
        |——USPS                     # USPS dataset
    |—— datasets
        |——minist.py                # Processing MNIST Datasets
        |——usps.py                  # Processing USPS Datasets
    |── model_utils
        |——config.py                # Processing configuration parameters
        |——device_adapter.py        # Get cloud ID
        |——local_adapter.py         # Get local ID
        |——moxing_adapter.py        # Parameter processing
    |── models
        |──discriminator.py         # Discriminator model Structure
        |──lenet.py                 # Lenet model Structure
    ├── scripts
        ├──run_eval.sh              # Shell script for evaluation on Ascend
        ├──run_eval_cpu.sh          # Shell script for evaluation on CPU
        ├──run_eval_gpu.sh          # Shell script for evaluation on GPU
        ├──run_train.sh             # Shell script for distributed training on Ascend
        ├──run_train_cpu.sh         # Shell script for training on CPU
        ├──run_train_gpu.sh         # Shell script for distributed training on GPU
    ├── sdefault_config.yaml        # Parameter configuration
    ├── eval.py                     # Evaluation script
    ├── export.py                   # Export checkpoint file to air/Mindir
    ├── README.md                   # English descriptions about ADDA
    ├── README_CN.md                # Chinese descriptions about ADDA
    ├── requirements.txt            # Required Package
    ├── train.py                    # Training script
    ├── uilts.py                    # Tool
   ```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for ADDA

  ```python  
  device_target:'Ascend'                                   # Device running the program
  dataset_mean: 0.5                                        # Normalized mean
  dataset_std: 0.5                                         # Normalized standard deviation
  batch_size:50                                            # Training batch size
  src_dataset: "MNIST"                                     # Source domain dataset
  src_encoder_checkpoint: "ADDA-source-encoder.ckpt"       # Source domain encoder weight file
  src_classifier_checkpoint: "ADDA-source-classifier.ckpt" # Source domain classifier weight file
  tgt_dataset: "USPS"                                      # Target domain dataset
  tgt_encoder_checkpoint: "ADDA-target-encoder.ckpt"       # Target domain encoder weight file
  model_root: "checkpoint"                                 # Weight file storage folder
  d_input_dims: 500                                        # Discriminator input layer dimension
  d_hidden_dims: 500                                       # Discriminator Hidden Layer Dimension
  d_output_dims: 2                                         # Discriminator output Layer Dimension
  d_model_checkpoint: "ADDA-critic.ckpt"                   # Discriminator weight file
  num_epochs_pre: 10                                       # Number of epochs trained in the source domain
  log_step_pre: 600                                        # Source domain record interval
  eval_step_pre: 2                                         # Source domain test interval
  save_step_pre: 2                                         # Source domain save interval
  num_epochs: 60                                           # Number of epochs trained in the target domain
  save_step: 10                                            # target domain save interval
  d_learning_rate: 3.0e-4                                  # Encoder learning rate
  c_learning_rate: 1.0e-4                                  # Discriminator learning rate
  beta1: 0.5                                               # The exponential decay rate of the first momentum matrix
  beta2: 0.9                                               # The exponential decay rate of the second momentum matrix
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
  Avg Accuracy = 97.027027%
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
|       Model Version        |                                             ADDA                                             |
|          Resource          |                              Ascend 910；CPU 24cores；Memory 96G;                              |
|       uploaded Date        |                                 27/05/2023 (month/day/year)                                  |
|     MindSpore Version      |                                            1.7.1                                             |
|          Dataset           |                           source domain：MNIST,target domain：USPS;                            |
|    Training Parameters     | num_epochs_pre=10,num_epochs=60,batch_size=50,d_learning_rate=3.0e-4,c_learning_rate=1.0e-4; |
|         Optimizer          |                                             Adam                                             |
|       Loss Function        |                                    Softmax Cross Entropy                                     |
|          outputs           |                                         probability                                          |
|           Speed            |                   1pc: 7 ms/step(source domain),79 ms/step(target domain);                   |
|         Total time         |                                       1pc: 22.78 mins                                        |
|       Parameters (M)       |                                             0.431                                             |
| Checkpoint for Fine tuning |                                      1.63M (.ckpt file)                                      |
|    Model for inference     |                            1.64M (.onnx file),  1.65M(.air file)                             |

### Evaluation Performance

#### Evaluation on USPS

| Parameters |              Ascend               |
|:-------:|:---------------------------------:|
| Model Version |               ADDA                |
| Resource  | Ascend 910；CPU 24cores；Memory 96G; |
| uploaded Date |    27/05/2023 (month/day/year)    |
| MindSpore Version |               1.7.1               |
| Dataset |         USPS,1860 images          |
| batch_size |                50                |
| outputs |            probability            |
| Accuracy |      1pc: 97.03%       |
| Model for inference |        1.64M (.onnx file)         |

# [Description of Random Situation](#contents)

In train.py, we use init_random_seed() function in utiles.py sets a random number seed.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
