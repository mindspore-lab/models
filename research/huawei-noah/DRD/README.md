# Decomposed Ranking Debiasing
Separating Examination and Trust Bias from Click Predictions for Unbiased Relevance Ranking

# Contents

- [Contents](#contents)
- [DRD Description](#DRD-description)
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

# [DRD Description](#contents)

we proposed a novel ULTR method called Decomposed Ranking Debiasing (DRD) to address the examination bias and trust bias in relevance ranking. Compared to existing propensity weighted methods, DRD decomposes each click prediction as a combination of a relevance term and other bias terms, and thus avoids involving the propensities in the denominator of labels. A joint learning algorithm is proposed to estimate the model parameters. Theoretical analysis showed DRD has the ability to learn unbiased relevance models with lower variances than existing methods. Groups of empirical studies also verified that DRD improved the baselines through effectively reducing the learning variances and accurately estimating the bias terms.

Separating Examination and Trust Bias from Click Predictions for Unbiased Relevance Ranking

WSDM 2023

# [Dataset](#contents)

- [YahooC14B](Olivier Chapelle and Yi Chang. 2011. Yahoo! learning to rank challenge overview. In Proceedings of the learning to rank challenge. PMLR, 1–24)
- [WEB30K](Tao Qin and Tie-Yan Liu. 2013. Introducing LETOR 4.0 datasets. arXiv preprint arXiv:1306.2597 (2013).)

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
  # Data preprocessing, Training and Evaluation on YahooC14B
  bash yahoo_main.sh
  # Data preprocessing, Training and Evaluation on YahooC14B
  bash web30k_main.sh
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

  ```text
  .
  └─DRD
    ├─README.md               # descriptions of DRD
    ├─main.py                 # run the training and evaluation of DRD
    ├─prepare_data.py         # preprocess the dataset
    ├─simulate_data.py        # simulate the semi-synthetic data
    ├─utils                   # some utils toolkit
      ├─__init__.py
      ├─arg_parser.py         # accept and parse the shell parameter
      ├─click_model.py        # click model for simulating the semi-synthetic data
      ├─data_warpper.py       # warp the dataset for training
      ├─early_stop.py         # early stop for obtain the best model
      ├─get_sparse.py         # get the sparse vector of feature
      ├─load_data.py          # load the raw dataset
      ├─metircs.py            # the metric for evaluating the performance
      ├─pairwise_trans.py     # transform the data to pairwise form
      ├─scale.py              # do the minmax scaling for web30k feature
      ├─set_seed.py           # setup the random seed
      ├─trans_format.py       # transform the column names when loading json file
    ├─train_alg               # training algorithm
      ├─__init__.py
      ├─DRD.py                # training algorithm of DRD
      ├─DRD_ideal.py          # training algorithm of DRD_ideal
    ├─simulate                # simulte semi-synthetic dataset
      ├─__init__.py
      ├─estimate_ips.py       # estimating the ips from data
      ├─estimate_rel.py       # estimating the relevance from data
      ├─simulate_click.py     # simulte the click
    ├─preprocessing           # preprocess the dataset
      ├─__init__.py
      ├─clean_data.py         # clean the raw dataset
      ├─output_json.py        # output the processed json files
      ├─rand_sample.py        # sample data for training initial ranker
      ├─svm_rank.py           # training initial SVM ranker
      ├─svm_rank_classify     # SVM rank classify files for Linux
      ├─svm_rank_classify.exe # SVM rank classify files for Windows
      ├─svm_rank_learn        # SVM rank learn files for Linux
      ├─svm_rank_learn.exe    # SVM rank learn files for Windows
    ├─models                  # ranking models
      ├─__init__.py
      ├─base_learn_alg.py     # base class of learning algorithm
      ├─evaluate.py           # evaluating the performance
      ├─loss_func.py          # loss functions
      ├─toy_model.py          # ranking models as the backbone of DRD
  ```

## [Script Parameters](#contents)

- Parameters that can be modified at the shell scripts

  ```text
  # Train
  fin: '../datasets/${dataname}/'                  # dataset path
  fout: '../datasets/${output_folder}/drd_ideal'   # output result path
  train_alg: 'drd_ideal'                           # training algorithm
  pairwise: 1                                      # is pairwise training
  lr: 1e-4                                         # learning rate
  weight_decay: 0                                  # weight decay
  batch_size: 128                                  # batch size
  topK: 10                                         # top-k cut-off in dataset
  epoch: 50                                        # total training epochs
  randseed: 61                                     # random seed
  drop_out: 0.1                                    # dropout ratio
  eta: 0.1                                         # severity of examination bias
  alpha: 0.1                                       # scale of rel leraning adjustment
  min_alpha: 0.1                                   # minimum scale of rel leraning adjustment
  beta : 0.1                                       # scale of bias leraning adjustment
  init_prob: 0.1                                   # initial probablity of trust bias
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters          | GPU                                                                                                                         |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Resource            | AMD Ryzen 2990WX 32-Core Processor;256G Memory;NVIDIA GeForce V100 16G                                                      |
| uploaded Date       | 08/31/2023 (month/day/year)                                                                                                 |
| MindSpore Version   | 2.0.0                                                                                                                       |
| Dataset             | YahooC14B                                                                                                                   |
| Training Parameters | epoch=50, batch_size=512, lr=1e-4, dropout=0.1, alpha=0.18                                                                  |
| Optimizer           | Adam                                                                                                                        |
| Training Alogoritm  | DRD_ideal                                                                                                                   |
| Outputs             | nDCG@1                                                                                                                      |
| Per Step Time       | 1.15s                                                                                                                       |
                                                                                          

# [Description of Random Situation](#contents)

- We set the random seed before preprocessing, training and evaluation

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models)


