# PMG

This is the official mindspore implementation of our paper: [PMG: Personalized Multimodal Generation with Large Language Models](https://arxiv.org/abs/2404.08677), Xiaoteng Shen, Rui Zhang, Xiaoyan Zhao, Jieming Zhu, Xi Xiao. In WWW 2024.

## Introduction

This paper proposes the first method for personalized multimodal generation using LLMs, showcases its applications and validates its performance via an extensive experimental study on two datasets.
The proposed method, Personalized Multimodal Generation (PMG for short) first converts user behaviors (e.g., clicks in recommender systems or conversations with a virtual assistant) into natural language to facilitate LLM understanding and extract user preference descriptions. Such user preferences are then fed into a generator, such as a multimodal LLM or diffusion model, to produce personalized content.

<center>
    <img src="https://cdn.jsdelivr.net/gh/mindspore-lab/models@master/research/huawei-noah/PMG/overview.png" width="70%" />
</center>

## Environment Requirements

+ Hardware
  + Ascend 910a
+ Framework
  + Python==3.9
  + MindSpore==2.2.1
+ Requirements
  + mindformers==0.8.0
  + gradio==3.48.0
  + numpy
  + pandas
+ For more information, please check the resources below：
  + MindSpore Tutorials (https://www.mindspore.cn/tutorials/en/master/index.html)
  + MindSpore Python API (https://www.mindspore.cn/docs/en/master/index.html)

## Quick Start

+ Step 1: Download dataset and checkpoints, and put them in the corresponding folders.
  
  + [MovieLens Latest Dataset (small)](https://grouplens.org/datasets/movielens/latest/)
  + [Llama2-7b-chat (mindspore)](https://github.com/mindspore-lab/mindformers/blob/master/docs/model_cards/llama2.md)
  + [Stable Diffusion v2.1 (mindspore)](https://github.com/mindspore-lab/mindone/tree/master/examples/stable_diffusion_v2)
  + [CLIP (mindspore)](https://github.com/XixinYang/CLIP-mindspore)

+ Step 2: Prepare the environment
  
  ```
  pip install -r requirements.txt
  ```

+ Step 3: Start the demo
  
  ```
  python main.py
  ```
  
  Here `main.py` lunch a front end server and it automatically call the back end api in `inference.py`

## Script and Sample Code

```
.
├── checkpoint_download
│   ├── clip                        # checkpoints of clip
│   └── llama2                      # checkpoints of llama2
├── clip                            # CLIP implemented by MindSpore
├── stable_diffusion_v2             # Stable Diffusion implemented by MindSpore
├── config                          # config of llama
├── data
│   ├── movie                       # Validation datasets
│   └── raw_data
│       └── ml-latest-small         # Dataset movielens
├── inference.py                    # Implement of PMG
├── main.css                        # Demo css
├── main.py                         # Demo
├── overview.png
├── PMG_utils.py
├── prompts.py                      # Prompts
├── readme.md
├── requirements.txt
└── soft_prompt.py                   # Implement of soft prompt
```
