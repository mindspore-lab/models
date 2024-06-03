### CLIP-mindspore
+ This folder contains code of CLIP implemented by MindSpore and is cloned from XixinYang/CLIP-mindspore (https://github.com/XixinYang/CLIP-mindspore).
+ To download checkpoints you can use this command:
```python
model, preprocess = clip.load("./ViT-B-32.ckpt", device="Ascend")
```
or manual download from:
```json
RN50: https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50-5d39bdab.ckpt,
RN101: https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN101-a9edcaa9.ckpt,
RN50x4: https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50x4-7b8cdb29.ckpt,
RN50x16: https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50x16-66ea7861.ckpt,
RN50x64: https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50x64-839951e0.ckpt,
ViT-B/32: https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_B_32-34c32b89.ckpt,
ViT-B/16: https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_B_16-99cbeeee.ckpt,
ViT-L/14: https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_L_14-1d8bde7f.ckpt,
ViT-L/14@336px: https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_L_14_336px-9ed46dee.ckpt,
```