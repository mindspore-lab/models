# Inference

## Installation

The code supports Ascend platform, here are some important dependencies are:

| mindspore-lite | mindspore | ascend driver | firmware | cann tookit/kernel |
| :------------: | :-------: | :-----------: | :------: | :----------------: |
| 2.3.1 | 2.3.1 | 24.1.RC2 | 7.3.t10 | 7.3.t10 |


## Convert MindSpore Weight File to Inference Weight File

Convert ckpt to mindir, please run: 

```shell
python export.py --checkpoint=your/path/to/ckpt
```

whereafter, two weight files  will be generated under the `./models` path by default,
`sam_vit_b.mindir` and `sam_vit_b_lite.mindir`.

See `python export.py --help` to explore more custom settings.

## Inference on Mindspore

```shell
python mindir_inference.py --model-path=./models/sam_vit_b.mindir --image-path=your/path/image.jpg
```

See `python mindir_inference.py --help` to explore more custom settings.

## Inference on Mindspore-Lite

```shell
python lite_inference.py --model-path=./models/sam_vit_b_lite.mindir --image-path=your/path/image.jpg
```

See `python lite_inference.py --help` to explore more custom settings.

## Performance

Experiments are tested on ascend310* with mindspore-lite 2.3.1 .

| model name | dataset | s/img |
| :--------: | :-----: | :---: |
| sam-vit-b | COCO2017 | 5.35 |
| sam-vit-b | FLARE22 | 5.35 |
