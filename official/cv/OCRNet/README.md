# OCRNet

[Object-Contextual Representations for Semantic Segmentation](https://arxiv.org/pdf/1909.11065)

## Introduction

OCRNet: object-contextual representations, characterizing a pixel by exploiting the representation of the corresponding object class,achieves competitive performance on various benchmarks: Cityscapes, ADE20K, LIP, PASCAL-Context and COCO-Stuff. 1st place on the
Cityscapes leaderboard on the ECCV 2020

## Requirement

- OCRNet is implemented based on  [mindspore 2.0](https://www.mindspore.cn/install/en) & [mindcv 0.2.2](https://github.com/mindspore-lab/mindcv)
- Install dependencies: pip install -r requirements.txt

## Data preparation
you need to download [cityscapes](https://www.cityscapes-dataset.com/)

your directory tree should be look like this:：

```text
└── Cityscapes
     ├── leftImg8bit
     |    ├── train
     |    ├── test
     |    ├── val
     └── gtFine
          ├── train
          ├── test
          └── val
```

## Performance

- npu : ascend 910A
- os: euler 2.8
- cpu: x86 with 96 cores
- mem: 756G
- dataset: cityscapes

| method | backbone | image size | cards | step | bs | fps | mIoU | mIoU(ms) | recipe/weights | 
| :-: | :-:| :-: | :-:| :-: | :-:| :-: | :-:| :-: | :-:
| OCRNet | [hrnet_w48](https://github.com/mindspore-lab/mindcv) | [1024, 2048] | 8 | 16k |  |  | | |[yaml](config/ocrnet/config_ocrnet_hrw48_16k.yml)/uploading
| OCRNet | [hrnet_w32](https://github.com/mindspore-lab/mindcv) | [1024, 2048] | 8 | 16k |  |  | | |[yaml](config/ocrnet/config_ocrnet_hrw32_16k.yml)/uploading

### Training

```shell
# single card
python train.py --config config/ocrnet/config_ocrnet_hrw32_16k.yaml
```
```shell
# multiple cards using openmpi
mpirun --allow-run-as-root -n 8 
    python train.py --config config/ocrnet/config_ocrnet_hrw32_16k.yaml
```

```shell
# training on modelarts
to do
```

### evaluation

```shell
# single card
python eval.py --config config/ocrnet/config_ocrnet_hrw32_16k.yaml --ckpt_path [downloaded_ckpt]

# multiple cards using openmpi
mpirun --allow-run-as-root -n 8 
    python eval.py --config config/ocrnet/config_ocrnet_hrw32_16k.yaml --ckpt_path [downloaded_ckpt]
```

### resume training

```shell
python train.py --config config/ocrnet/config_ocrnet_hrw32_16k.yaml --resume_ckpt [ckpt_to_resume]
```

## Citation
```sheel
@article{YuanCW20,
  title={Object-Contextual Representations for Semantic Segmentation},
  author={Yuhui Yuan and Xilin Chen and Jingdong Wang},
  booktitle={ECCV},
  year={2020}
}
```


