# Contents

- [Contents](#contents)
- [SparCode Description](#SparCode-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
  - [Script and Sample Code](#script-and-sample-code)
  - [Script Parameters](#script-parameters)
  - [Performance](#performance)
    - [Training Performance](#training-performance)
    - [Inference Performance](#inference-performance)

# [SparCode Description](#contents)

Two-tower models are a prevalent matching framework for recommendation, which have been widely deployed in industrial applications. The success of two-tower matching attributes to its efficiency in retrieval among a large number of items, since the item tower can be precomputed and used for fast Approximate Nearest Neighbor (ANN) search. However, it suffers two main challenges, including limited feature interaction capability and reduced accuracy in online serving. Existing approaches attempt to design novel late interactions instead of dot products, but they still fail to support complex feature interactions or lose retrieval efficiency. To address these challenges, we propose a new matching paradigm named SparCode, which supports not only sophisticated feature interactions but also efficient retrieval. Specifically, SparCode introduces an all-to-all interaction module to model fine-grained query-item interactions. Besides, we design a discrete code-based sparse inverted index jointly trained with the model to achieve effective and efficient model inference. Extensive experiments have been conducted on open benchmark datasets to demonstrate the superiority of our framework. The results show that SparCode significantly improves the accuracy of candidate item matching while retaining the same level of retrieval efficiency with two-tower models.

- Liangcai Su, Fan Yan, Jieming Zhu, Xi Xiao, Haoyi Duan, Zhou Zhao, Zhenhua Dong, Ruiming Tang. [Beyond Two-Tower Matching: Learning Sparse Retrievable Cross-Interactions for Recommendation](https://dl.acm.org/doi/abs/10.1145/3539618.3591643). In SIGIR 2023.

# [Dataset](#contents)

We test the code on the ml-1m datasets:
- train data: data/ml-1m/data.csv
- test data: data/ml-1m/test.csv

# [Environment Requirements](#contents)

- Framework
  - [MindSpore-2.2.0](https://www.mindspore.cn/install/en)
- Requirements
  - pandas
  - numpy
  - logging
  - mindspore==2.2.0
  - tqdm
  - sklearn
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r2.2/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/r2.2/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on GPU
  
  ```
  # run training and evaluation
  python work.py
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
└─Sparcode
  ├─README.md             # descriptions of SparCode
  ├─models                # different models
    ├─__init__.py
    ├─BaseModel.py        # base model
    ├─DQMatch.py          # model
    ├─layers.py           # 
    ├─vq                  # vq model
        ├─__init__.py
        ├─vq_embedding.py # vq_embedding model
        ├─vq.py           # vq model            
  ├─data.csv              # train dataset
  ├─test.csv              # test dataset
  ├─metric.py             #  
  ├─load_csv_dataset.py   # load dataset
  └─work.py               # Run File
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be found in `work.py`
examples:model,batch_size,nference_batch_size ,embedding_size

## [Performance](#contents)

### Training Performance And Inference Performance

| Parameters          | GPU                              |
| ------------------- | -------------------------------- |
| Model Version       | SparCode                         |
| Resource            | GPU                              |
| uploaded Date       | ml-1m                            |
| MindSpore Version   | 2.2.0                            |
| Training Parameters | epoch=10, batch_size=64, lr=1e-4 |
| Optimizer           | Adam                             |
| Loss Function       | softmax_loss + weight * mse      |

ModelZoo Homepage
Note: This model will be move to the /models/research/recommend

Please check the official homepage.
