# Contents

- [Contents](#contents)
    - [LPCNet Description](#lpcnet-description)
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
    - [Inference Process](#inference-process)
        - [Generate input data for network](#generate-input-data-for-network)
        - [Export MindIR](#export-mindir)
        - [Result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)

<!-- /TOC -->

## [LPCNet Description](#contents)

LPCNet is a lowbitrate neural vocoder based on linear prediction and sparse recurrent networks.

[Article](https://jmvalin.ca/papers/lpcnet_codec.pdf): J.-M. Valin, J. Skoglund, A Real-Time Wideband Neural Vocoder at 1.6 kb/s Using LPCNet, Proc. INTERSPEECH, arxiv:1903.12087, 2019.

## [Model Architecture](#contents)

LPCNet has two parts: frame rate network and sample rate newtwork. Frame rate network consists of two convolutional layers and two fully connected layers, this network extracts features from 5-frames context. Sample rate network consists of two GRU layers and dual fully connected layer (modification of fully connected layer). The first GRU layer waights are sparcified. Sample rate network gets features extracted by frame rate network along with linear prediction for current timestep, sample end excitation for previous timestep and predicts current excitation via sigmoid function and binary tree probability representation.

## [Dataset](#contents)

Dataset used: [ESC-50](<https://github.com/karolpiczak/ESC-50?tab=readme-ov-file#download>)

- Dataset size：600MB
    - Divide 10% of the data as the test set and the rest as the training set.
- Data format：binary files
    - Note：The ESC-50 dataset is a labeled collection of 2000 environmental recordings, consisting of 5-second records. The network reconstructs the original audio from these quantified features.

- Download the dataset (only .wav files are needed), the directory structure is as follows:

    ```train-clean-100
    ├─README.md
    ├─LICENSE
    ├─pytest.ini
    ├─requirements.txt
    ├─audio
    └─meta
    ├─audio
        ├─1-137-A-32.wav
        ├─1-977-A-39.wav
        ├─1-1791-A-26.wav
        ...
        └─5-263902-A-36.wav
    ├─meta
        ├─esc50.csv
        └─esc50-human.xlsx
    ```

## [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
# enter script dir, compile code for feature quantization
bash run_compile.sh

# generate training data (sox should be installed sequentially before running the command)
bash run_process_train_data.sh [TRAIN_DATASET_PATH] [OUTPUT_PATH]
# example: bash run_process_train_data.sh ./dataset_path/train ~/dataset_path/training_dataset/

# train LPCNet
bash run_standalone_train_ascend.sh [PREPROCESSED_TRAINING_DATASET_PATH] [CHECKPOINT_SAVE_PATH]
# example: bash run_standalone_train_ascend.sh ~/dataset_path/training_dataset/ ./ckpt/

# generate test data (10 files are selected from test-clean for evaluation)
bash run_process_eval_data.sh [EVAL_DATASET_PATH] [OUTPUT_PATH]
# example: bash run_process_eval_data.sh ./dataset_path/test_dataset ~/dataset_path/test_features/

# evaluate LPCNet
bash run_eval_ascend.sh [TEST_DATASET_PATH] [OUTPUT_PATH] [CHECKPOINT_SAVE_PATH]
# example: bash run_eval_ascend.sh ~/dataset_path/test_features/ ./eval_results/ ./ckpt/lpcnet-4_37721.ckpt
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── audio
    ├── LPCNet
        ├── README.md                          // description of LPCNet in English
        ├── requirements.txt                   // required packages
        ├── scripts
        │   ├──run_compile.sh                  // compile code feature extraction and quantization code
        │   ├──run_infer_310.sh                // inference
        │   ├──run_proccess_train_data.sh      // generate training dataset from .wav files
        |   ├──run_process_eval_data.sh        // generate eval dataset from .wav files
        │   ├──run_stanalone_train_ascend.sh   // train in Ascend
        ├── src
        │   ├──rnns                            // dynamic GRU implementation
        │   ├──dataloader.py                   // dataloader for model training
        │   ├──lossfuncs.py                    // loss function
        │   ├──lpcnet.py                       // lpcnet implementation
        │   ├──mdense.py                       // dual fully connected layer implementation
        |   └──ulaw.py                         // u-law qunatization
        ├── third_party                        // feature extraction and quantization (C++)
        ├── ascend310_infer                    // for inference on Ascend (C++)
        ├── train.py                           // train in Ascend main program
        ├── process_data.py                    // generate training dataset from KITTI .bin files main program
        ├── eval.py                            // evaluation main program
        ├──export.py                           // exporting model for infer
```

### [Script Parameters](#contents)

```text
Major parameters in train.py：
features: Path to features
data: Path to 16-bit PCM aligntd with features
output: Path where .ckpt stored
--batch-size：Training batch size
--epochs：Total training epochs
--device：Device where the code will be implemented. Optional values are "Ascend", "CPU"
--checkpoint：The path to the checkpoint file saved after training.（recommend ）

Major parameters in eval.py：
test_data_path: path to test dataset，test data is features extracted and quantized by run_process_eval_data.sh
output_path: The path where decompressed / reconstructed files stored
model_file: The path to the checkpoint file which needs to be loaded
```

### [Training Process](#contents)

#### Training

- Running on Ascend

  ```bash
  python train.py [FEATURES_FILE] [AUDIO_DATA_FILE] [CHECKPOINT_SAVE_PATH] --device=[DEVICE_TARGET] --batch-size=[batch-size]
  # or enter script dir, run 1P training script
  bash run_stanalone_train_ascend.sh ~/dataset_path/training_dataset/ ./ckpt/
  ```

  After training, the loss value will be achieved as follows:

  ```bash
  epoch: 1 step: 5208, loss is 3.528735876083374
  ...
  epoch: 4 step: 37721, loss is 3.1588170528411865
  ...
  ```

  The model checkpoint will be saved in the specified directory.

### [Evaluation Process](#contents)

#### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

  ```bash
  python eval.py [TEST_DATA_PATH] [OUTPUT_PATH] [CHECKPOINT_SAVE_PATH]
  # or enter script dir, run evaluation script
  bash run_eval_ascend.sh [TEST_DATASET_PATH] [OUTPUT_PATH] [CHECKPOINT_SAVE_PATH]
  ```

## [Inference Process](#contents)

### Generate input data for network

```shell
# Enter script dir, run run_process_data.sh script
bash run_process_eval_data.sh [EVAL_DATASET_PATH] [OUTPUT_PATH]
# example: bash run_process_eval_data.sh ./dataset_path/test_dataset ~/dataset_path/test_features/
```

### Export MindIR

```shell
python export.py --ckpt_file=[CKPT_PATH] --max_len=[MAX_LEN] --out_file=[OUT_FILE]
# Example:
python export.py --ckpt_file='./checkpoint/ms-4_37721.ckpt'  --out_file=lpcnet --max_len 500
# NOTE: max_len is the max number of 10 ms frames which can be processed, audios longer will be truncated
```

The ckpt_file parameter is required

### Result

Inference result is saved in ./infer.log.

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters          | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| Network Name | LPCNet                                                   |
| Resource  | Ascend 910；CPU 191 core；Memory 755G;                            |
| Uploaded Date | TBD                                                          |
| MindSpore Version | 1.8.1                                                        |
| Dataset | 6.16 GB of clean speech                               |
| Training Parameters | epoch=4, batch_size=64 , lr=0.001 |
| Optimizer | Adam                                                         |
| Loss Function | SparseCategoricalCrossentropy                                          |
| Output   | distribution                                         |
| Loss      | 3.15881                                                          |
| Speed     | 400ms/step             |
| Total Time | 8.5h                               |
| parameters(M) | 1.2M                                                        |
| Checkpoint for Fine tuning | 20.51M (.ckpt file)                                               |
| Scripts   | [lpcnet script](https://gitee.com/mindspore/models/tree/master/official/audio/LPCNet) |

### Inference Performance

| Parameters        | Ascend                                                       |
| ----------------- | ------------------------------------------------------------ |
| Network Name      | LPCNet                                                   |
| Resource          | Ascend 910                                                   |
| Uploaded Date     | TBD                                                          |
| MindSpore Version | 1.8.1                                                        |
| Dataset           | Feature files constructed from 10 files in ESC50 test                                     |
| batch_size        | 1                                                            |
| Output            | Reconstructed 16-bit PCM audio|
| Accuracy         | 0.024 (MSE)|