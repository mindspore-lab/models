# ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation

## Introduction
This is the mindspore implementation of ***ReLLa*** proposed in the paper [ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation](https://arxiv.org/abs/2308.11131).

We also provide a pytorch implementation in this [repo](https://github.com/LaVieEnRose365/ReLLa).

## Requirements
- Python==3.9
- MindSpore==2.2.1
- mindformers==0.8.0

## Data preprocess
You can directly use the processed data from [this link](https://drive.google.com/drive/folders/1av6mZpk0ThmkOKy5Y_dUnsLRdRK8oBjQ?usp=sharing). (including data w/o and w/ retrieval: full testing set, sampled training set, history length 30/30/60 for Ml-1m/Ml-25m/BookCrossing).

Or you can preprocess by yourself.
Scripts for data preprocessing of [BookCrossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/), [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/), [MovieLens-25M](https://grouplens.org/datasets/movielens/25m/) are included in this [repo](https://github.com/LaVieEnRose365/ReLLa).

## Get semantic embeddings
Get semantic item embeddings for retrieval.
~~~python
python encode.py --item_dir PATH_1 --embed_dir PATH_2 --ckpt_dir PATH_3
~~~

## Retrieval and construct prompts
~~~python
python retrieval.py --embed_dir PATH_1 --data_dir PATH_2 --indice_dir PATH_3 --meta_dir PATH_4 --output_dir PATH_5 --use_pca --K 30 --temp_type sequential
~~~
Note that you can control the history length used and template type by *K* and *temp_type*.


## Inference
~~~python
python inference.py --data_dir PATH_1 --ckpt_dir PATH_2 --eval_batch_size 1
~~~