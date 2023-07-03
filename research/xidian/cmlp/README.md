# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [ConvMLP Description](#ConvMLP-description)
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
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Export Process](#export-process)
        - [Export](#export)
    - [Inference Process](#inference-process)
        - [Inference](#inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
            - [CMLP train on CIFAR-10](#CMLP-train-on-cifar-10)
            - [CMLP train on CIFAR-100](#CMLP-train-on-cifar-100)
       

# [ConvMLP Description](#contents)

On 2021.9.18, UO&UIUC proposed ConvMLP: a hierarchical convolutional MLP for visual recognition. as a light-weight, stage-wise, co-design with convolutional layers and MLPs, ConvMLP was developed on ImageNet-1k with Only 2.4G MACs and 9M parameters 
(15% and 19% of MLP-Mixer-B/16, respectively) to achieve 76.8% Top-1 accuracy. The article was accepted by CVPR in 2023.

[Paper](https://openaccess.thecvf.com/content/CVPR2023W/WFM/html/Li_ConvMLP_Hierarchical_Convolutional_MLPs_for_Vision_CVPRW_2023_paper.html)：Li J, Hassani A, Walton S, et al. Convmlp: Hierarchical convolutional mlps for vision[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 6306-6315.

# [Model Architecture](#contents)

To address the constraints on the input dimension in the framework of MLP, the authors first replace all spatial mlps with cross-channel connections , and build a pure MLP baseline model. To compensate for the spatial information interaction, the authors add a lightweight convolutional stage to the remaining MLP stages , and use the convolutional layers for downsampling . In addition, to increase the spatial connectivity of the MLP stages, the authors add a 3 × 3 deep convolution between the two channel MLPs in each MLP block , hence the name Conv-MLP block. The authors prototyped the ConvMLP model for image classification by co-designing the convolutional and MLP layers.
# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)

- Dataset size：175M，60,000 32*32 colorful images in 10 classes
    - Train：146M，50,000 images  
    - Test：29M，10,000 images
- Data format：binary files
    - Note：Data will be processed in src/dataset.py

Dataset used: [CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)

- Dataset size：161M，60,000 32*32 colorful images in 100 classes
    - Train：132M，50,000 images  
    - Test：29M，10,000 images
- Data format：binary files
    - Note：Data will be processed in src/dataset.py

# [Features](#contents)

## Mixed Precision

The training method using [mixed precision](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html) uses supported single-precision and half-precision data to increase the training speed of deep learning neural networks, while maintaining the network accuracy that can be achieved with single-precision training. Mixed-precision training improves computational speed and reduces memory usage while supporting the training of larger models or achieving larger batches on specific hardware.
As an example, if the input data type is FP16 operator, MindSpore backend will automatically reduce the precision to process the data if the input data type is FP32. Users can open the INFO log and search for "reduce precision" to see the reduced precision operators.
# [Environment Requirements](#contents)

- Hardware（Ascend/CPU）
    - Prepare hardware environment with Ascend/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend

  ```yaml
  # Add data set path, take training cifar10 as an example
  train_data_path:./data/cifar-10-batches-bin
  val_data_path:./data/cifar-10-val-bin

  # Add checkpoint path parameters before inference
  chcekpoint_path:./checkpoint/cifa10_CMLP.ckpt
  ```

  ```python
  # run training example
  python train.py

  # run evaluation example
  python evl.py 
   ```

  In order to run on the CPU side, you need to change the device_target parameter in train.py to CPU

 

- running on CPU

  ```python
  # run training example
  python train.py
  # example:parser.add_argument('--device_target', type=str, default='Ascend')->
  # parser.add_argument('--device_target', type=str, default='CPU')

  # run evaluation example
   python evl.py 
   ```



# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md // All model-related descriptions
    ├── CMLP
        ├── README.md // CMLP-related instructions
        ├─ checkpoint
        │ ├──cifa10_CMLP.ckpt // cifa10 mindspore pre-training model
        │ ├──cifa100_CMLP.pth // cifa10 CMLP official pre-training model
        │ ├──cifa100_CMLP.ckpt // cifa100 mindspore pre-training model
        │ ├──cifa10_CMLP.pth // cifa100 CMLP official pre-training model
        ├── data // data folder
        │ ├──cifar-10-batches-bin
        │ ├──cifar-100-binary    
        ├── model
        │ ├── mlp.py
        ├── train.py // training script
        ├── evl.py // evaluation script
        ├── dataset.py // data processing script
        ├── pth2ckpt.py // pass the pytorch pre-training model parameters into the mindspore model parameters
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for CMLP, CIFAR-10 dataset

```python
  'run_modelarts':'False' # whether to use modelart
  'is_distributed':False # whether to distribute training
  'device_id':4 # Select training device
  'batch_size':64 # training batch size
  'epoch_size':125 # total number of training epochs
  'dataset_choose':cifar10 # dataset selection
  'device_target':Ascend # hardware selection
  'save_checkpoint_path':. /ckpt" # Model save address 
  
  ```

- config for CMLP, CIFAR-100 dataset

  ```python
  'run_modelarts':'False' # whether to use modelart
  'is_distributed':False # whether to distribute training
  'device_id':4 # Select training device
  'batch_size':64 # training batch size
  'epoch_size':125 # total number of training epochs
  'dataset_choose':cifar100 # dataset selection
  'device_target':Ascend # hardware selection
  'save_checkpoint_path':. /ckpt" # Model save address 
  
  ```
## [Export Process](#contents)

### [Export](#content)

The parameters from the official CMLP pytorch pre-training model need to be put into the mindspore model before training.

**Note**: The relative address of the pytotch model needs to be given in pth2ckpt.py, and the parameter sizes of the CMLP model need to correspond one to the other.
```shell
python pth2ckpt.py
```

## [Training Process](#contents)

### Training

- running on Ascend

  ```python
  python train.py
  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```bash
  # grep "loss is " train.log
  epoch:1 step:768, loss is 0.96960
  epcoh:2 step:768, loss is 0.82834
  ...
  ```

  The model checkpoint will be saved in the current directory.



  After training, you'll get some checkpoint files under the folder `./ckpt_0/` by default.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on CIFAR-10 dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/CMLP/train_CMLP_cifar10-125_390.ckpt".

  ```python
  python eval.py
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```bash
  accuracy:{'acc':0.9806}
  ```

  Note that for evaluation after distributed training, please set the checkpoint_path to be the last saved checkpoint file such as "username/CMLP/train_parallel0/train_CMLP_cifar10-125_48.ckpt". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "accuracy: " eval.log
  accuracy: {'acc': 0.9217}
  ```

- evaluation on CIFAR-10 dataset when running on GPU

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/CMLP/train/ckpt_0/train_CMLP_cifar10-125_390.ckpt".

  ```python
  python eval.py --checkpoint_path=[CHECKPOINT_PATH] > eval.log 2>&1 &  
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "accuracy: " eval.log
  accuracy: {'acc': 0.930}
  ```

  OR,

  ```bash
  bash run_eval_gpu.sh [CHECKPOINT_PATH]
  ```

  The above python command will run in the background. You can view the results through the file "eval/eval.log". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "accuracy: " eval/eval.log
  accuracy: {'acc': 0.930}
  ```



# [Model Description](#contents)

## [Performance](#contents)

### Inference Performance

#### CMLP train on CIFAR-10

| Parameters | Ascend | GPU |
| ------------------- |----------------------|-----------------|
| model_version | convmlp_s | convmlp_s |
| Resources | Ascend 910; System ubuntu | GPU |
| upload date | 2023-07-05 | 2023-07-05 |
| MindSpore version | 1.8.1 | 1.8.1 |
| dataset | CIFAR-10, 10,000 images | CIFAR-10, 10,000 images |
| batch_size | 128 | 128 |
| Output | Probability | Probability |
| Accuracy | Single card: 98.06%; | Single card: 97.64% |
#### CMLP train on CIFAR-100

| Parameters | Ascend | GPU |
| ------------------- |----------------------|------------------|
| model_version | convmlp_s | convmlp_s |
| Resources | Ascend 910; System ubuntu | GPU |
| upload date | 2023-07-05 | 2023-07-05 |
| MindSpore version | 1.8.1 | 1.8.1 |
| dataset | CIFAR-100, 50,000 images | CIFAR-100, 50,000 images |
| batch_size | 128 | 128 |
| Output | Probability | Probability |
| accuracy | single card: 85.12%; | single card: 85.48% |
