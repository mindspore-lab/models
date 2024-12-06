# Inference

## Installation

The code supports Ascend platform, here are some important dependencies are:

| mindspore | ascend driver | firmware | cann tookit/kernel |
| :-------: | :-----------: | :------: | :----------------: |
| 2.3.1 | 24.1.RC2 | 7.3.t10 | 7.3.t10 |


## Convert MindSpore Weight File to Inference Weight File

Convert ckpt to mindir, please run: 

```shell
python export.py --checkpoint=your/path/to/ckpt
```

whereafter, one weight file  will be generated under the `./models` path by default,
`sam_vit_b.mindir`.

See `python export.py --help` to explore more custom settings.

## Inference on Mindspore

```shell
python mindir_inference.py --model-path=./models/sam_vit_b.mindir --image-path=your/path/image.jpg
```

See `python mindir_inference.py --help` to explore more custom settings.
