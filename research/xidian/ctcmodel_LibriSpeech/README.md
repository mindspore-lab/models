# Contents

- [Contents](#contents)
- [Model Description](#model-description)
  - [Paper](#paper)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
  - [Script and Sample Code](#script-and-sample-code)
  - [Script Parameters](#script-parameters)
  - [Data Preprocessing](#data-preprocessing)
  - [Training Process](#training-process)
    - [Usage](#usage)
      - [Running on Ascend](#running-on-ascend)
  - [Evaluation Process](#evaluation-process)
    - [Usage](#usage-1)
      - [Running on Ascend](#running-on-ascend-1)
- [Model Description](#model-description-1)
  - [Performance](#performance)
    - [Evaluation Performance](#evaluation-performance)
    - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)

# [Model Description](#contents)

CTCModel uses the CTC criterion to train the RNN model to complete the morpheme labeling task. The full name of CTC is
Connectionist Temporal Classification, and the Chinese name is "Connection Time Series Classification". This method
mainly solves the problem that the neural network label and output are not aligned. Supervise the label sequence to
train. CTC is widely used in speech recognition, OCR and other tasks, and has achieved remarkable results.

## Paper

[Paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf): Alex Graves, Santiago Fernández, Faustino J. Gomez, Jürgen
Schmidhuber:
"Connectionist temporal classification": labelling unsegmented sequence data with recurrent neural networks. ICML 2006:
369-376

# [Model Architecture](#contents)

The model includes a two-layer bidirectional LSTM model with an input dimension of 39, which is the dimension of the
extracted speech features, a fully connected layer with an output dimension of 62, the number of labels + 1, and 61
represents a blank symbol.

# [Dataset](#contents)

The dataset used is: [Mini LibriSpeech](<https://www.openslr.org/31/>), which includes two formats of FLAC, TXT
Preprocessing of the downloaded and decompressed data:

- Read raw data, convert voice data to wav format
- Read raw data, split text data into but data txt tags
- Read voice data and tag data, extract voice signal features through mfcc and second-order difference
- Fill in the processed data and convert the processed data into MindRecord format
- The data preprocessing script preprocess_data.sh is provided here, which will be described in detail in the data
  preprocessing script section later.
- The length of the training set after preprocessing is 4620, and the length of the test set is 1680

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend, GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.html)
- See the `requirements.txt` file, the usage is as follows:

```bash
pip install -r requirements.txt
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
CTCModel
├── scripts
│ ├── eval.sh                       # launch Ascend evaluation
│ └── train_alone.sh                # launch Ascend standalone training
├── src
│ ├── dataset.py                    # data preprocessing
│ ├── eval_callback.py              # evaluation callback while training
│ ├── loss.py                       # custom loss function
│ ├── metric.py                     # custom metrics
│ ├── model.py                      # backbone model
│ ├── model_for_eval.py             # custom network evaluation
│ ├── model_for_train.py            # custom network training
│ └── model_utils
│   ├── config.py                   # parse configuration document
│   ├── device_adapter.py           # distinguish local /modelarts files
│   ├── __init__.py
│   ├── local_adapter.py            # obtain device information for local training
│   └── moxing_adapter.py           # modelarts configuration, exchange files
├── default_config.yaml             # parameters configuration file
├── eval.py                         # network evaluation
├── export.py                       # export MINDIR format
├── preprocess_Libri.py             # Libri_min_data preprocessing
├── preprocess_data.py              # data preprocessing
└── train.py                        # training script
```

## [Script Parameters](#contents)

The relevant parameters of data preprocessing, training, and evaluation are in the `default_config.yaml` file.

- Data preprocessing related parameters

```text
dataset_dir     # The directory where the MindRecord files obtained by preprocessing are saved
train_dir       # The directory of the original training data before preprocessing
test_dir        # The directory of the original test data before preprocessing
train_name      # The name of the preprocessed training MindRecord file
test_name       # The preprocessed Test the name of the MindRecord file
```

- Model related parameters

```text
feature_dim         # Input feature dimension, which is consistent with the preprocessed data dimension, 39
batch_size          # batch size
hidden_size         # hidden layer dimension
n_class             # number of labels, dimension of the final output of the model, 62
n_layer             # LSTM layer number
max_sequence_length # maximum sequence length, all sequences are padded to This length, 1555
max_label_length    # The maximum length of the label, all labels are padded to this length, 75
```

- Training related parameters

```text
train_path              # Training set MindRecord file
test_path               # Test set MindRecord file
save_dir                # Directory where the model is saved
epoch                   # Number of iteration rounds
lr_init                 # Initial learning rate
clip_value              # Gradient clipping threshold
save_check              # Whether to save the model
save_checkpoint_steps   # The number of steps to save the model
keep_checkpoint_max     # The maximum number of saved models
train_eval              # Whether to train while testing
interval                # How many steps to test
run_distribute          # Is distributed training
dataset_sink_mode       # Whether data sinking is enabled

```

- Evaluation related parameters

```text
test_path         # Test set MindRecord file
checkpoint_path   # Model save path
test_batch_size   # Test set batch size
beam              # Greedy decode (False) or prefix beam decode (True), the default is greedy decode
```

- Export related parameters

```text
file_name   # export file name
file_format # export file format, MINDIR
```

- Configure related parameters

```text
enable_modelarts  # Whether training on modelarts, default: False
device_target     # Ascend or GPU
device_id         # Device number
```

## [Data Preprocessing](#contents)

Before data preprocessing, please make sure to install the `python-speech-features` library and run the example:
First, run the preprocess_Libri.py file to preprocess the Libri_min data.

```bash
python preprocess_data.py \
       --dataset_dir ./dataset \
       --train_dir /data/Libri_mini/TRAIN \
       --test_dir /data/Libri_mini/TEST \
       --train_name train.mindrecord \
       --test_name test.mindrecord
```

```text
Parameters:
    --dataset_dir The path to store the processed MindRecord file, the default is ./dataset, it will be automatically created
    --train_dir The directory where the original training set data is located
    --test_dir The directory where the original test set data is located
    --train_name The name of the training file generated, the default is train.mindrecord
    --test_name The name of the generated test file, the default is test.mindrecord
    Other parameters can be set through the default_config.yaml file
```

Or you can run the script:

```bash
bash scripts/preprocess_data.sh [DATASET_DIR] [TRAIN_DIR] [TEST_DIR]
```

All three parameters are required, respectively corresponding to the above `--dataset_dir`, `--train_dir`, `--test_dir`.

Data preprocessing process is slow, it takes about ten minutes.

## [Training Process](#contents)

### Usage

#### Running on Ascend

- Standalone training

Run the example:

```bash
python train.py \
       --train_path ./dataset/train.mindrecord0 \
       --test_path ./dataset/test.mindrecord0 \
       --save_dir ./save \
       --epoch 120 \
       --train_eval True \
       --interval 5 \
       --device_id 0 > train.log 2>&1 &
```

```text
parameters:
    --train_path training set file path
    --test_path test set file path
    --save_dir model save path
    --epoch iteration rounds
    --train_eval whether to test while training
    --interval Test interval
    --device_id device number
    Other parameters can be set through the default_config.yaml file
```

Or you can run the script:

```bash
bash scripts/train_alone.sh [TRAIN_PATH] [TEST_PATH] [SAVE_DIR] [DEVICE_ID]
```

All four parameters are required, corresponding to the above `--train_path`, `--test_path`, `--save_dir`, `--device_id`.

Commands will run in the background, you can view the results through `train.log`.

The first epoch operator takes a long time to compile, about 60 minutes, and each epoch after that is about 7 minutes.

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

Make sure to install the edit-distance library before evaluating and run the example:

```bash
python eval.py \
       --test_path ./dataset/test.mindrecord0 \
       --checkpoint_path ./save/best.ckpt \
       --beam False \
       --device_id 0 > eval.log 2>&1 &
```

```text
parameters:
    --test_path test Set file path
    --checkpoint_path path to load model
    --device_id device number
    --beam greedy decoding or prefix beam decoding
    Other parameters can be set through the default_config.yaml file
```

Or you can run the script:

```bash
bash scripts/eval.sh [TEST_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
```

The 3 parameters are all required, respectively corresponding to the above `--test_path`, `--checkpoint_path`,
`--device_id`.

The above command runs in the background, you can view the results through `eval.log`, The test results are as follows

```text
READ:{'read': 0.3038}
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance


| parameters                 | CTCModel                                                      |
| -------------------------- | ---------------------------------------------------------------|
| resource                   | Ascend910             |
| Upload Date              | To be determined                                    |
| MindSpore version           | 1.8.1                                                          |
| Data set                    | Mini LibriSpeech，Training set length: 1519                                                 |
| Training parameter       | epoch=300, batch_size = 64, lr_init=0.01,clip_value=5.0   |
| Optimizer                  | Adam                                                           |
| Loss function              | CTCLoss                                |
| Output                    | LER                                                   |
| Loss value                       | 717.1                                                        |
| Running speed                      | 7229.011 ms/step                                   |
| Total training time       | about one day;                                                                             |

### Inference Performance

| parameters                 | CTCModel                                                      |
| -------------------------- | ----------------------------------------------------------------|
| resource                   | Ascend910                   |
| Upload Date               | To be determined                                 |
| MindSpore version          | 1.8.1                                                           |
| Data set                   | Mini LibriSpeech，Test Set Size:1089                         |
| batch_size                 | 1                                                               |
| Output                   | LER:0.9                       |

# [Description of Random Situation](#contents)

Randomness comes from the following two main sources.

- Parameter initialization
- Rotation of the data set
