# Contents

- [Contents](#contents)
- [ReLoop2 Description](#reloop2-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
  - [Script and Sample Code](#script-and-sample-code)
  - [Script Parameters](#script-parameters)

# [ReLoop2 Description](#contents)

ReLoop2 is a self-correcting learning loop that facilitates fast model adaptation in online recommender systems through responsive error compensation. Inspired by the slow-fast complementary learning system observed in human brains, we propose an error memory module that directly stores error samples from incoming data streams. These stored samples are subsequently leveraged to compensate for model prediction errors during testing, particularly under distribution shifts. The error memory module is designed with fast access capabilities and undergoes continual refreshing with newly observed data samples during the model serving phase to support fast model adaptation.

Jieming Zhu, Guohao Cai, Junjie Huang, Zhenhua Dong, Ruiming Tang, Weinan Zhang. [ReLoop2: Building Self-Adaptive Recommendation Models via Responsive Error Compensation Loop](https://arxiv.org/abs/2306.08808). In KDD 2023.

# [Model Architecture](#contents)
The base model is DeepFM which consists of two components. The FM component is a factorization machine, which is proposed in to learn feature interactions for recommendation. The deep component is a feed-forward neural network, which is used to learn high-order feature interactions. The FM and deep component share the same input raw feature vector, which enables DeepFM to learn low- and high-order feature interactions simultaneously from the input raw features.

# [Dataset](#contents)

- [Criteo Kaggle Display Advertising Challenge Dataset](http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz)

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend, GPU, or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- preprocess dataset

  ```bash
  #download dataset
  #Please refer to [Criteo Kaggle Display Advertising Challenge Dataset] to get the download URL and assign it to `DATA_LINK`
  mkdir -p data/origin_data && cd data/origin_data
  wget DATA_LINK
  tar -zxvf dac.tar.gz

  #preprocess dataset
  python -m src.preprocess_data  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
  ```

- running on Ascend

  ```shell
  # run training example
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target=Ascend \
    --do_eval=True > ms_log/output.log 2>&1 &

  # run distributed training example
  bash scripts/run_distribute_train.sh 8 /dataset_path /rank_table_8p.json

  # run evaluation example
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target=Ascend > ms_log/eval_output.log 2>&1 &
  OR
  bash scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/deepfm.ckpt
  ```


- running on GPU

  ```shell
  # run training example
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target=GPU \
    --do_eval=True > ms_log/output.log 2>&1 &

  # run distributed training example
  bash scripts/run_distribute_train_gpu.sh 8 /dataset_path

  # run evaluation example
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target=GPU > ms_log/eval_output.log 2>&1 &
  OR
  bash scripts/run_eval.sh 0 GPU /dataset_path /checkpoint_path/deepfm.ckpt
  ```

- running on CPU

  ```shell
  # run training example
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target=CPU \
    --do_eval=True > ms_log/output.log 2>&1 &

  # run evaluation example
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target=CPU > ms_log/eval_output.log 2>&1 &
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```path
.
└─deepfm
  ├─README.md
  ├─default_config.yaml               # default config
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in Ascend or GPU
    ├─run_distribute_train.sh         # launch distributed training(8p) in Ascend
    ├─run_distribute_train_gpu.sh     # launch distributed training(8p) in GPU
    └─run_eval.sh                     # launch evaluating in Ascend or GPU
  ├─src
    ├─model_utils
     ├─__init__.py
     ├─config.py
     ├─device_target.py
     ├─local_adapter.py
     └─moxing_adapter.py
    ├─__init__.py                     # python init file
    ├─callback.py                     # define callback function
    ├─deepfm.py                       # deepfm network
    ├─dataset.py                      # create dataset for deepfm
    └─preprocess_data.py              # data preprocess
  ├─eval.py                           # eval net
  └─train.py                          # train net
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- train parameters

  ```help
  optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Dataset path
  --ckpt_path CKPT_PATH
                        Checkpoint path
  --eval_file_name EVAL_FILE_NAME
                        Auc log file path. Default: "./auc.log"
  --loss_file_name LOSS_FILE_NAME
                        Loss log file path. Default: "./loss.log"
  --do_eval DO_EVAL     Do evaluation or not. Default: True
  --device_target DEVICE_TARGET
                        Ascend or GPU. Default: Ascend
  ```

- eval parameters

  ```help
  optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        Checkpoint file path
  --dataset_path DATASET_PATH
                        Dataset path
  --device_target DEVICE_TARGET
                        Ascend or GPU. Default: Ascend
  ```
