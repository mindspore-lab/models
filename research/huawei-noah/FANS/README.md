# Contents

- [Contents](#contents)
- [FANS Description](#FANS-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
  - [Script and Sample Code](#script-and-sample-code)
  - [Script Parameters](#script-parameters)
  - [Performance](#performance)

# [FANS Description](#contents)

Recently, Transformer-based models have shown promise in comprehending contextual information and capturing item reltionships in a list. However, deploying them in real-time industrial applications is challenging, mainly because the autoregressive generation mechanism used in them is time-consuming. In this paper, we propose a novel fast non-autoregressive sequence generation model, namely FANS, to enhance inference efficiency and quality for item list continuation. 

- Qijiong Liu, Jieming Zhu, Jiahao Wu, Tiandeng Wu, Zhenhua Dong, Xiao-Ming Wu. [FANS: Fast Non-Autoregressive Sequence Generation for Item List Continuation](https://arxiv.org/abs/2304.00545). In WWW 2023.

# [Dataset](#contents)

- train data: ./data/stContUni/aotm-n10/train_all.csv 
- test data: ./data/stContUni/aotm-n10/test_all.csv 

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
  python work.py --config aotm-bert-double-n10.yaml --exp curriculum-bert-double-step.yaml
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
├─config
├─data
│  └─ListContUni
│      └─aotm-n10
│          └─clusters
├─exp
├─loader
│  ├─init
│  │  ├─bert_init.py               #bert init
│  │  └─model_init.py              #model init
│  ├─task
│  │  ├─bert
│  │  │  └─curriculum_cluster_mlm_task.py  
│  │  ├─utils
│  │  │  ├─base_classifiers.py             #base 
│  │  │  ├─base_cluster_mlm_task.py        #base task
│  │  │  ├─base_curriculum_mlm_task.py
│  │  │  └─base_mlm_task.py
│  │  ├─base_batch.py          
│  │  ├─base_loss.py       
│  │  ├─base_task.py
│  │  └─task_manager.py
│  ├─data.py         
│  └─model_dataloader.py
├─saving
│  └─aotm-n10
│      └─BERT-E64
│          └─curriculum-bert-double-step
├─utils
│  ├─config_initializer.py       
│  └─metric.py
│  └─splitter.py
├─load_mindspore_csv.py    #load dataset
├─model_bert.py            #model
├─README.md
├─tinybert.py              #tinbert model
└─work.py                  #main file
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be found in `work.py`

- --config  model configuration file
- --exp     train configuration file
- --cuda    GPU device

## [Performance](#contents)

### Training Performance And Inference Performance

| Parameters          | GPU                              |
| ------------------- | -------------------------------- |
| Model Version       | FANS                             |
| Resource            | GPU                              |
| uploaded Date       | ListContUni                      |
| MindSpore Version   | 2.2.0                            |
| Training Parameters | epoch=10, batch_size=64, lr=1e-4 |
| Optimizer           | Adam                             |
