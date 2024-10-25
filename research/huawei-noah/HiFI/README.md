# Contents

- [Contents](#contents)
- [Description](#Description)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Sample Code Structure](#sample-code-structure)
    - [Script Parameters](#script-parameters)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Description](#contents)

This is the sample code of the paper ``HiFI: Hierarchical Fairness-aware Integrated Ranking with Constrained Reinforcement Learning'' (WWW'23).

Abstract: Integrated ranking is a critical component in industrial recommendation platforms. It combines candidate lists from different upstream channels or sources and ranks them into an integrated list, which will be exposed to users. During this process, to take responsibility for channel providers, the integrated ranking system needs to consider the exposure fairness among channels, which directly affects the opportunities of different channels being displayed to users. 
Besides, personalization also requires the integrated ranking system to consider the user's diverse preference on different channels besides items.
Existing methods are hard to address both problems effectively. In this paper, we propose a Hierarchical Fairness-aware Integrated ranking (HiFI) framework. It contains a channel recommender and an item recommender, and the fairness constraint is on channels with constrained RL. We also design a gated attention layer (GAL) to effectively capture users' multi-faceted preferences.
We compare HiFI with various baselines on public and industrial datasets, and HiFI achieves the state-of-the-art performance on both utility and fairness metrics. We also conduct an online A/B test to further validate the effectiveness of HiFI.


# [Environment Requirements](#contents)

- Hardware CPU and GPU.
    - Prepare hardware environment with CPU processor or Ascend GPU.
- Framework
    - [MindSpore-2.2.14](https://www.mindspore.cn/install/en)
- Requirements
  - asttokens       2.4.1
  - astunparse      1.6.3
  - Bottleneck      1.3.5
  - certifi         2022.12.7
  - mindspore       2.2.14
  - mkl-fft         1.3.1
  - mkl-random      1.2.2
  - mkl-service     2.4.0
  - numexpr         2.7.3
  - numpy           1.21.6
  - packaging       24.0
  - pandas          1.3.5
  - Pillow          9.5.0
  - pip             22.3.1
  - protobuf        4.24.4
  - psutil          5.9.8
  - python-dateutil 2.8.2
  - pytz            2022.7
  - scipy           1.7.3
  - setuptools      65.6.3
  - six             1.16.0
  - wheel           0.38.4

  
- For more information, please check the resources below.
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation our sample code as follows:

  ```shell
  python main.py --use_rcpo --use_gal
  ```
# [Script Description](#contents)

## [Sample Code Structure](#contents)

```text 
.
|-HiFI
  |-README.md             # descriptions of HiFI
  |-Dataset.py            # The sample dataset of HiFI model.
  |-main.py               # The main train and evaluate code of HiFI model.
  |-model.py              # The channel recommender of HiFI.
  |-model_item.py         # The item recommender of HiFI.
  |-data                  # The data folder.
    |- inter_seq.csv      # The sample user-item interaction data.
    |- test_inter_seq.csv # The sample test user-item interaction data.
    |- user_seq.csv       # The sample user feature data.
    |- item_seq.csv       # The sample item feature data.
  
```
## [Script Parameters](#contents)

    path args:
    --data_dir: the path of data
    --result_dir: the path of result

    data args:
    --item_nums: total number of items
    --channel_nums: total number of channels
    --dense_dim: the dimension of dense feature
    --user_nums: total number of users
    --user_hist_len: the length of user history
    --seq_len: the length of interaction sequence
    --seed: random seed
    --device: device type

    train args:
    --batch_size: train batch size
    --epoch_num: train epoch number
    --learning_rate: learning rate
    --tau: soft update parameter
    --soft_update_iter: soft update interval
    --item_step_len: the step length of item recommender
    --eval_interval: evaluation interval

    model args:
    --emb_dim: embedding dimension
    --mlp_dim: mlp dimension
    --keep_prob: 1.-dropout rate
    --gamma: reward discount factor
    --use_rcpo: whether to use rcpo
    --crr_type: the type of crr
    --critic_weight: the weight of critic loss
    --use_gal: whether to use gal
    --num_heads: the number of heads in attention

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models)
