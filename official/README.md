### [Computer Vision]

#### Image Classification
Accuracies are reported on ImageNet-1K

| model | acc@1 | bs | num_cards | ms/step | amp | device | config | 
:-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| vgg11| 71.86 | 32 | 8 |  61.63  |  O2 |  Ascend_910A  | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg) | 
| vgg13| 72.87 | 32 | 8 |  66.47  |  O2 |  Ascend_910A  | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg) |
| vgg16| 74.53 | 32 | 8 |  73.68  |  O2 |  Ascend_910A  | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg) |
| vgg19| 75.21 | 32 | 8 |  81.13  |  O2 |  Ascend_910A  | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg) |
| resnet18| 70.21 | 32 | 8 |  23.98  |  O2 |  Ascend_910A  | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet) |
| resnet34| 74.15 | 32 | 8 |  23.98  |  O2 |  Ascend_910A  | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet) |
| resnet50| 76.69 | 32 | 8 |  31.97  |  O2 |  Ascend_910A  | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet) |
| resnet101| 78.24 | 32 | 8 | 50.76   |  O2 |  Ascend_910A  | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet) |
| resnet152| 78.72 | 32 | 8 |  70.94  |  O2 |  Ascend_910A  | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet) |
| resnetv2_50| 76.90 | 32 | 8 | 35.72   |  O2 |  Ascend_910A  | [mindcv_resnetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnetv2) |
| resnetv2_101| 78.48 | 32 | 8 |  56.02  |  O2 |  Ascend_910A  | [mindcv_resnetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnetv2) |
| dpn92  | 79.46 | 32 | 8 | 79.89   |  O2 |  Ascend_910A  | [mindcv_dpn](https://github.com/mindspore-lab/mindcv/blob/main/configs/dpn) |
| dpn98  | 79.94 | 32 | 8 | 106.60  |  O2 |  Ascend_910A  | [mindcv_dpn](https://github.com/mindspore-lab/mindcv/blob/main/configs/dpn) |
| dpn107 | 80.05 | 32 | 8 | 107.60  |  O2 |  Ascend_910A  | [mindcv_dpn](https://github.com/mindspore-lab/mindcv/blob/main/configs/dpn) |
| dpn131 | 80.07 | 32 | 8 | 143.57  |  O2 |  Ascend_910A  | [mindcv_dpn](https://github.com/mindspore-lab/mindcv/blob/main/configs/dpn) |
| densenet121  | 75.64 | 32 | 8 | 48.07   |  O2 |  Ascend_910A  | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet) |
| densenet161  | 79.09 | 32 | 8 | ??  |  O2 |  Ascend_910A  | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet) |
| densenet169 | 77.26 | 32 | 8 | 73.14  |  O2 |  Ascend_910A  | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet) |
| densenet201 | 78.14 | 32 | 8 | 96.12  |  O2 |  Ascend_910A  | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet) |
| seresnet18 | 71.81 | 64 | 8 | 50.39  |  O2 |  Ascend_910A  | [mindcv_senet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet) |
| seresnet34 | 75.36 | 64 | 8 | 50.54 |  O2 |  Ascend_910A  | [mindcv_senet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet) |
| seresnet50 | 78.31 | 64 | 8 | 98.37  |  O2 |  Ascend_910A  | [mindcv_senet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet) |
| seresnext26 | 77.18 | 64 | 8 | 73.72  |  O2 |  Ascend_910A  | [mindcv_senet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet) |
| seresnext50 | 78.71 | 64 | 8 | 113.82  |  O2 |  Ascend_910A  | [mindcv_senet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet) |