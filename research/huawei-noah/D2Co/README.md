# Debiased and Denoised Watch Time Correction (D2Co)
Uncovering User Interest from Biased and Noised Watch Time in Video Recommendation

# Contents

- [Contents](#contents)
- [D2Co Description](#D2Co-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [D2Co Description](#contents)

 we propose a model called Debiased and Denoised watch time Correction (D2Co) to mitigate the duration bias and noisy watching. Specifically, we propose to regard the distribution of watch time in each duration length as a mixture of latent bias and noise distributions. A duration-wise Gaussian mixture model is employed to estimate the parameters of these latent distributions. Since the adjacent value of duration should have similar properties, a frequency-weighted moving average is used to smooth the estimated bias and noise parameters sequence. Then we utilized a sensitivity-controlled correction function to separate the user interest from the watch time, which is robust to the estimation error of bias and noise parameters.

Uncovering User Interest from Biased and Noised Watch Time in Video Recommendation

Recsys 2023

# [Dataset](#contents)

- [KuaiRand](http://kuairand.com/)
- [WeChat](https://algo.weixin.qq.com/)

# [Environment Requirements](#contents)

- Hardware（CPU and GPU）
    - Prepare hardware environment with CPU processor and GPU of Nvidia.
- Framework
    - [MindSpore-2.0.0](https://www.mindspore.cn/install/en)
- Requirements
  - numpy
  - tqdm
  - pandas
  - argparse
  - skit-learn
  - mindspore==2.0.0

- For more information, please check the resources below:
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start data preprocessing, training and evaluation as follows:

- Data preprocessing, Training and Evaluation

  ```shell
  # Data preprocessing, Training and Evaluation on KuaiRand
  bash kuairand_main.sh
  # Data preprocessing, Training and Evaluation on WeChat
  bash wechat.sh
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

  ```text
  .
  └─D2Co
    ├─README.md               # descriptions of D2Co
    ├─main.py                 # run the training and evaluation of D2Co
    ├─prepare_data.py         # preprocess the dataset
    ├─train_model.py          # training recommendation model
    ├─utils                   # some utils toolkit
      ├─__init__.py
      ├─arg_parser.py         # accept and parse the shell parameter
      ├─data_warpper.py       # warp the dataset for training
      ├─early_stop.py         # early stop for obtaining the best model
      ├─evaluation.py         # evaluating the performance
      ├─metrics.py            # the metric for evaluating the performance
      ├─moving_avg.py         # moving average for smoothing
      ├─set_seed.py           # setup the random seed
      ├─summary_dat.py        # filtrating feature and calculating feature dim
    ├─preprocessing           # preprocess the dataset
      ├─__init__.py
      ├─cal_baseline_label.py # calculating the baseline label
      ├─cal_gmm_label.py      # calculating the D2Co label
      ├─cal_ground_truth.py   # calculating the ground truth label
      ├─pre_kuairand.py       # preprocessing kuairand dataset
      ├─pre_wechat.py         # preprocessing wechat dataset
    ├─models                  # recommendation models
      ├─__init__.py
      ├─DeepFM.py             # DeepFM backbone model

  ```

## [Script Parameters](#contents)

- Parameters that can be modified at the shell scripts

  ```text
  # Train
  dat_name                                         # dataset for training and evaluation
  model_name                                       # backbone model
  label_name                                       # training label
  fout: '../rec_datasets/${output_folder}/'        # output result path
  randseed: 61                                     # random seed
  lr: 1e-3                                         # learning rate
  weight_decay: 0                                  # weight decay
  batch_size: 512                                  # batch size
  epoch_num: 50                                    # total training epochs
  randseed: 61                                     # random seed
  drop_out: 0.0                                    # dropout ratio
  patience: 5                                      # waiting patience for early stop
  alpha: -0.05                                     # sensitivity control term
  windows_size:3                                   # Windows size of moving average
  group_num                                        # Groups of D2Q
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters          | GPU                                                                                                                         |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Resource            | AMD Ryzen 2990WX 32-Core Processor;256G Memory;NVIDIA GeForce V100 16G                                                      |
| uploaded Date       | 12/13/2023 (month/day/year)                                                                                                 |
| MindSpore Version   | 2.0.0                                                                                                                       |
| Dataset             | KuaiRand                                                                                                                    |
| Training Parameters | epoch=50, batch_size=512, lr=5e-4, dropout=0.0, alpha=-0.05, windows_size=3, randseed=61                                    |
| Optimizer           | Adam                                                                                                                        |
| Training label      | D2Co                                                                                                                        |
| Outputs             | nDCG@1                                                                                                                      |
| Per Step Time       | 63.2s                                                                                                                       |
                                                                                          

# [Description of Random Situation](#contents)

- We set the random seed before preprocessing, training and evaluation

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models)


