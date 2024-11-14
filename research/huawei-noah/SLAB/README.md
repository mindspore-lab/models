# SLAB

This is an official mindspore implementation of our paper "**SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized BatchNorm**". In this paper, we investigate the computational bottleneck modules of efficient transformer, i.e., normalization layers and attention modules. Layer normalization is commonly used in transformer architectures but is not computational friendly due to statistic calculation during inference. However, replacing Layernorm with more efficient batch normalization in transformer often leads to inferior performance and collapse in training. To address this problem, we propose a novel method named PRepBN to progressively replace LayerNorm with re-parameterized BatchNorm in training. During inference, the proposed PRepBN could be simply re-parameterized into a normal BatchNorm, thus could be fused with linear layers to reduce the latency. Moreover, we propose a simplified linear attention (SLA) module that is simply yet effective to achieve strong performance. Extensive experiments on image classification as well as object detection demonstrate the effectiveness of our proposed method. For example, powered by the proposed methods, our SLAB-Swin obtains 
83.6% top-1 accuracy on ImageNet with 16.2ms latency, which is 2.4ms less than that of Flatten-Swin with 0.1 higher accuracy.

## Data preparation

The ImageNet dataset should be prepared as follows:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

## Inference

Please download the mindspore checkpoint from [here](https://github.com/xinghaochen/SLAB/releases/download/ckpts/swin_tiny_prepbn_mindspore.ckpt) and evaluate the model:

```shell
python eval.py --device_id 0 --device_target GPU --swin_config ./src/configs/swin_tiny_patch4_window7_224.yaml --pretrained ./swin_tiny_prepbn_mindspore.ckpt
```

## Reference
If you find SqueezeTime useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:
```
@article{guo2024slab,
  title={SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized BatchNorm},
  author={Guo, Jialong and Chen, Xinghao and Tang, Yehui  and Wang, Yunhe},
  journal={arXiv preprint},
  year={2024}
}
```
