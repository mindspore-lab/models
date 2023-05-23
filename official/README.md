### [Computer Vision]

#### Image Classification
Accuracies are reported on ImageNet-1K

| model | acc@1 | bs | cards | ms/step | amp | device | config | 
:-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| vgg11| 71.86 | 32 | 8 |  61.63  |  O2 |  Ascend_910A  | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg) | 
| vgg13| 72.87 | 32 | 8 |  66.47  |  O2 |  Ascend_910A  | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg) |
| vgg16| 74.61 | 32 | 8 |  73.68  |  O2 |  Ascend_910A  | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg) |
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
| densenet161  | 79.09 | 32 | 8 | 115.11  |  O2 |  Ascend_910A  | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet) |
| densenet169 | 77.26 | 32 | 8 | 73.14  |  O2 |  Ascend_910A  | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet) |
| densenet201 | 78.14 | 32 | 8 | 96.12  |  O2 |  Ascend_910A  | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet) |
| seresnet18 | 71.81 | 64 | 8 | 50.39  |  O2 |  Ascend_910A  | [mindcv_senet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet) |
| seresnet34 | 75.36 | 64 | 8 | 50.54 |  O2 |  Ascend_910A  | [mindcv_senet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet) |
| seresnet50 | 78.31 | 64 | 8 | 98.37  |  O2 |  Ascend_910A  | [mindcv_senet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet) |
| seresnext26 | 77.18 | 64 | 8 | 73.72  |  O2 |  Ascend_910A  | [mindcv_senet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet) |
| seresnext50 | 78.71 | 64 | 8 | 113.82  |  O2 |  Ascend_910A  | [mindcv_senet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet) |
| skresnet18 | 73.09 | 64 | 8 | coming  |  O2 |  Ascend_910A  | [mindcv_sknet](https://github.com/mindspore-lab/mindcv/blob/main/configs/sknet) |
| skresnet34 | 76.71 | 32 | 8 | 43.96  |  O2 |  Ascend_910A  | [mindcv_sknet](https://github.com/mindspore-lab/mindcv/blob/main/configs/sknet) |
| skresnet50_32x4d | 79.08 | 64 | 8 | 65.95  |  O2 |  Ascend_910A  | [mindcv_sknet](https://github.com/mindspore-lab/mindcv/blob/main/configs/sknet) |
| resnext50_32x4d | 78.53 | 32 | 8 | 50.25  |  O2 |  Ascend_910A  | [mindcv_resnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnext) |
| resnext101_32x4d | 79.83 | 32 | 8 | 68.85  |  O2 |  Ascend_910A  | [mindcv_resnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnext) |
| resnext101_64x4d | 80.30 | 32 | 8 | 112.48  |  O2 |  Ascend_910A  | [mindcv_resnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnext) |
| resnext152_64x4d | 80.52 | 32 | 8 | 157.06  |  O2 |  Ascend_910A  | [mindcv_resnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnext) |
| rexnet_x09 | 77.07 | 64 | 8 | 145.08 |  O2 |  Ascend_910A  | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet) |
| rexnet_x10 | 77.38 | 64 | 8 | 156.67 |  O2 |  Ascend_910A  | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet) |
| rexnet_x13 | 79.06 | 64 | 8 | 203.04 |  O2 |  Ascend_910A  | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet) |
| rexnet_x15 | 79.94 | 64 | 8 | 231.41 |  O2 |  Ascend_910A  | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet) |
| rexnet_x20 | 80.64 | 64 | 8 | 308.15 |  O2 |  Ascend_910A  | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet) |
| resnest50 | 80.81 | 128 | 8 | 376.18 |  O2 |  Ascend_910A  | [mindcv_resnest](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnest) |
| resnest101 | 82.50 | 128 | 8 | 719.84 |  O2 |  Ascend_910A  | [mindcv_resnest](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnest) |
| res2net50 | 79.35 | 32 | 8 | 49.16 |  O2 |  Ascend_910A  | [mindcv_res2net](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net) |
| res2net101 | 79.56 | 32 | 8 | 49.96 |  O2 |  Ascend_910A  | [mindcv_res2net](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net) |
| res2net50_v1b | 80.32 | 32 | 8 | 93.33 |  O2 |  Ascend_910A  | [mindcv_res2net](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net) |
| res2net101_v1b | 95.41 | 32 | 8 | 86.93 |  O2 |  Ascend_910A  | [mindcv_res2net](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net) |
| googlenet | 72.68 | 32 | 8 | 23.26 |  O0 |  Ascend_910A  | [mindcv_googlenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/googlenet) |
| inceptionv3 | 79.11 | 32 | 8 | 49.96 |  O0 |  Ascend_910A  | [mindcv_inceptionv3](https://github.com/mindspore-lab/mindcv/blob/main/configs/inceptionv3) |
| inceptionv4 | 80.88 | 32 | 8 | 93.33 |  O0 |  Ascend_910A  | [mindcv_inceptionv4](https://github.com/mindspore-lab/mindcv/blob/main/configs/inceptionv4) |
| mobilenet_v1_025 | 53.87 | 64 | 8 | 75.93 |  O2 |  Ascend_910A  | [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1) |
| mobilenet_v1_050 | 65.94 | 64 | 8 | 51.96 |  O2 |  Ascend_910A  | [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1) |
| mobilenet_v1_075 | 70.44 | 64 | 8 | 57.55 |  O2 |  Ascend_910A  | [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1) |
| mobilenet_v1_100 | 72.95 | 64 | 8 | 44.04 |  O2 |  Ascend_910A  | [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1) |
| mobilenet_v2_075 | 69.98 | 256 | 8 | wait |  O3 |  Ascend_910A  | [mindcv_mobilenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2) |
| mobilenet_v2_100 | 72.27 | 256 | 8 | wait |  O3 |  Ascend_910A  | [mindcv_mobilenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2) |
| mobilenet_v2_140 | 75.56 | 256 | 8 | wait |  O3 |  Ascend_910A  | [mindcv_mobilenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2) |
| mobilenet_v3_small | 68.10 | 75 | 8 | wait |  O3 |  Ascend_910A  | [mindcv_mobilenetv3](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv3) |
| mobilenet_v3_large | 75.23 | 75 | 8 | wait |  O3 |  Ascend_910A  | [mindcv_mobilenetv3](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv3) |
| shufflenet_v1_g3_x0_5 | 57.05 | 64 | 8 | 142.69 |  O0 |  Ascend_910A  | [mindcv_shufflenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv1) |
| shufflenet_v1_g3_x1_5 | 67.77 | 64 | 8 | 267.79 |  O0 |  Ascend_910A  | [mindcv_shufflenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv1) |
| shufflenet_v2_x0_5 | 57.05 | 64 | 8 | 142.69 |  O0 to confirm|  Ascend_910A  | [mindcv_shufflenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv1) |
| shufflenet_v2_x1_0 | 67.77 | 64 | 8 | 267.79 |  O0 |  Ascend_910A  | [mindcv_shufflenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv1) |
| shufflenet_v2_x1_5 | 57.05 | 64 | 8 | 142.69 |  O0 |  Ascend_910A  | [mindcv_shufflenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv1) |
| shufflenet_v2_x2_0 | 67.77 | 64 | 8 | 267.79 |  O0 |  Ascend_910A  | [mindcv_shufflenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv1) |
| hrnet_w32 | 80.64 | 128 | 8 | 335.73 |  O2 |  Ascend_910A  | [mindcv_hrnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/hrnet) |
| hrnet_w48 | 81.19 | 128 | 8 | 463.63 |  O2 |  Ascend_910A  | [mindcv_hrnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/hrnet) |
| ghostnet_50 | 66.03 | 128 | 8 | 220.88 |  O3 |  Ascend_910A  | [mindcv_ghostnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/ghostnet) |
| ghostnet_100 | 73.78 | 128 | 8 | 222.67 |  O3 |  Ascend_910A  | [mindcv_ghostnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/ghostnet) |
| ghostnet_130 | 75.50 | 128 | 8 | 223.11 |  O3 |  Ascend_910A  | [mindcv_ghostnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/ghostnet) |
| nasnet_a_4x1056 | 73.65 | 256 | 8 | 1562.35 |  O0 |  Ascend_910A  | [mindcv_nasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/nasnet) |
| mnasnet_0.5 | 68.07 | 512 | 8 | 367.05 |  O3 |  Ascend_910A  | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet) |
| mnasnet_0.75 | 71.81 | 256 | 8 | 151.02 |  O0 |  Ascend_910A  | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet) |
| mnasnet_1.0 | 74.28 | 256 | 8 | 153.52 |  O0 |  Ascend_910A  | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet) |
| mnasnet_1.4 | 76.01 | 256 | 8 | 194.90 |  O0 |  Ascend_910A  | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet) |
| efficientnet_b0 | 76.89 | 128 | 8 | 276.77 |  O2 |  Ascend_910A  | [mindcv_efficientnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/efficientnet) |
| efficientnet_b1 | 78.95 | 128 | 8 | 435.90 |  O2 |  Ascend_910A  | [mindcv_efficientnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/efficientnet) |
| regnet_x_200mf| 68.74 | 64 | 8 | 47.56 |  O2 |  Ascend_910A  | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet) |
| regnet_x_400mf| 73.16 | 64 | 8 | 47.56 |  O2 |  Ascend_910A  | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet) |
| regnet_x_600mf| 73.34 | 64 | 8 | 48.36 |  O2 |  Ascend_910A  | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet) |
| regnet_x_800mf| 76.04 | 64 | 8 | 47.56 |  O2 |  Ascend_910A  | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet) |
| regnet_y_200mf| 70.30 | 64 | 8 | 58.35 |  O2 |  Ascend_910A  | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet) |
| regnet_y_400mf| 73.91 | 64 | 8 | 77.94 |  O2 |  Ascend_910A  | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet) |
| regnet_y_600mf| 75.69 | 64 | 8 | 79.94 |  O2 |  Ascend_910A  | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet) |
| regnet_y_800mf| 76.52 | 64 | 8 | 81.93 |  O2 |  Ascend_910A  | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet) |
| mixnet_s | 75.52 | 128 | 8 | 340.18 |  O3 |  Ascend_910A  | [mindcv_mixnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mixnet) |
| mixnet_m | 76.64 | 128 | 8 | 384.68 |  O3 |  Ascend_910A  | [mindcv_mixnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mixnet) |
| mixnet_l | 78.73 | 128 | 8 | 389.97 |  O3 |  Ascend_910A  | [mindcv_mixnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mixnet) |
| bit_resnet50 | 76.81 | 32 | 8 | 130.60 |  O0 |  Ascend_910A  | [mindcv_bit](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit) |
| bit_resnet50x3 | 80.63 | 32 | 8 | 533.09 |  O0 |  Ascend_910A  | [mindcv_bit](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit) |
| bit_resnet101 | 77.93| 16 | 8 | 128.15 |  O0 |  Ascend_910A  | [mindcv_bit](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit) |
| repvgg_a0 | 72.19 | 32 | 8 | 27.63 |  O0 |  Ascend_910A  | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg) |
| repvgg_a1 | 74.19 | 32 | 8 | 27.45 |  O0 |  Ascend_910A  | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg) |
| repvgg_a2 | 76.63 | 32 | 8 | 39.79 |  O0 |  Ascend_910A  | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg) |
| repvgg_b0 | 74.99 | 32 | 8 | 33.05 |  O0 |  Ascend_910A  | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg) |
| repvgg_b1 | 78.81 | 32 | 8 | 68.88 |  O0 |  Ascend_910A  | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg) |
| repvgg_b2 | 79.29 | 32 | 8 | 106.90 |  O0 |  Ascend_910A  | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg) |
| repvgg_b3 | 80.46 | 32 | 8 | 137.24 |  O0|  Ascend_910A  | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg) |
| repvgg_b1g2 | 78.03 | 32 | 8 | 59.71 |  O2 |  Ascend_910A  | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg) |
| repvgg_b1g4 | 77.64 | 32 | 8 | 65.83 |  O2 |  Ascend_910A  | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg) |
| repvgg_b2g4 | 78.80 | 32 | 8 | 89.57 |  O2 |  Ascend_910A  | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg) |
| repmlp_t224 | 76.71 | 128 | 8 | 973.88 |  O2 |  Ascend_910A  | [mindcv_repmlp](https://github.com/mindspore-lab/mindcv/blob/main/configs/repmlp) |
| convnext_tiny | 81.91 | 128 | 8 | 343.21 |  O2 |  Ascend_910A  | [mindcv_convnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnext) |
| convnext_small | 83.40 | 128 | 8 | 405.96 |  O2 |  Ascend_910A  | [mindcv_convnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnext) |
| convnext_base | 83.32 | 128 | 8 | 531.10 |  O2 |  Ascend_910A  | [mindcv_convnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnext) |
| vit_b_32_224 | 75.86 | 256 | 8 | ?? |  O2 |  Ascend_910A  | [mindcv_vit](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit) |
| vit_l_16_224 | 76.34| 48 | 8 | ?? |  O2 |  Ascend_910A  | [mindcv_vit](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit) |
| vit_l_32_224 | 73.71 | 128 | 8 | ?? |  O2 |  Ascend_910A  | [mindcv_vit](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit) |
| swin_tiny | 80.82 | 256 | 8 | 1765.65 |  O2 |  Ascend_910A  | [mindcv_swintransformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/swintransformer) |
| pvt_tiny | 74.81 | 128 | 8 | 310.74 |  O2 |  Ascend_910A  | [mindcv_pvt](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt) |
| pvt_small | 79.66 | 128 | 8 | 431.15 |  O2 |  Ascend_910A  | [mindcv_pvt](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt) |
| pvt_medium | 81.82 | 128 | 8 | 613.08 |  O2 |  Ascend_910A  | [mindcv_pvt](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt) |
| pvt_large | 81.75 | 128 | 8 | 860.41 |  O2 |  Ascend_910A  | [mindcv_pvt](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt) |
| pvt_v2_b0 | 71.50 | 128 | 8 | 338.78 |  O2 |  Ascend_910A  | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2) |
| pvt_v2_b1 | 78.91 | 128 | 8 | 337.94 |  O2 |  Ascend_910A  | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2) |
| pvt_v2_b2 | 81.99 | 128 | 8 | 503.79 |  O2 |  Ascend_910A  | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2) |
| pvt_v2_b3 | 82.84 | 128 | 8 | 738.90 |  O2 |  Ascend_910A  | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2) |
| pvt_v2_b4 | 83.14 | 128 | 8 | 1030.06 |  O2 |  Ascend_910A  | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2) |
| pit_ti | 72.96 | 128 | 8 | 339.44 |  O2 |  Ascend_910A  | [mindcv_pit](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit) |
| pit_xs | 78.41 | 128 | 8 | 338.70 |  O2 |  Ascend_910A  | [mindcv_pit](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit) |
| pit_s | 80.56 | 128 | 8 | 336.08 |  O2 |  Ascend_910A  | [mindcv_pit](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit) |
| pit_b | 81.87 | 128 | 8 | 350.33 |  O2 | 7 Ascend_910A  | [mindcv_pit](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit) |
| coat_lite_tiny | 77.35 | 64 | 8 | 258.07 |  O2 |  Ascend_910A  | [mindcv_coat](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat) |
| coat_lite_mini | 78.51 | 64 | 8 | 265.44 |  O2 |  Ascend_910A  | [mindcv_coat](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat) |
| coat_tiny | 79.67 | 64 | 8 | 580.50 |  O2 |  Ascend_910A  | [mindcv_coat](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat) |
| convit_tiny | 73.66 | 256 | 8 | 200.00 |  O2 |  Ascend_910A  | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit) |
| convit_tiny_plus | 77.00 | 256 | 8 | 214.50 |  O2 |  Ascend_910A  | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit) |
| convit_small | 81.63 | 192 | 8 | 381.50 |  O2 |  Ascend_910A  | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit) |
| convit_small_plus | 81.80 | 192 | 8 | 422.10 |  O2 |  Ascend_910A  | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit) |
| convit_base | 82.10 | 128 | 8 | 658.90 |  O2 |  Ascend_910A  | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit) |
| convit_base_plus | 81.96 | 128 | 8 | 399.70 |  O2 |  Ascend_910A  | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit) |
| crossvit_9 | 73.56 | 256 | 8 | 685.25 |  O3 |  Ascend_910A  | [mindcv_crossvit](https://github.com/mindspore-lab/mindcv/blob/main/configs/crossvit) |
| crossvit_15 | 81.08 | 256 | 8 | 1086.00 |  O3 |  Ascend_910A  | [mindcv_crossvit](https://github.com/mindspore-lab/mindcv/blob/main/configs/crossvit) |
| crossvit_18 | 81.93 | 256 | 8 | 1037.60 |  O3 |  Ascend_910A  | [mindcv_crossvit](https://github.com/mindspore-lab/mindcv/blob/main/configs/crossvit) |
| visformer_tiny | 78.28 | 128 | 8 | 300.70 |  O3 |  Ascend_910A  | [mindcv_visformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer) |
| visformer_tiny_v2 | 78.82 | 256 | 8 | 602.50 |  O3 |  Ascend_910A  | [mindcv_visformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer) |
| visformer_small | 81.76 | 64 | 8 | 155.90 |  O3 |  Ascend_910A  | [mindcv_visformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer) |
| visformer_small_v2 | 82.17 | 64 | 8 | 153.10 |  O3 |  Ascend_910A  | [mindcv_visformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer) |
| edgenext_xx_small | 71.02 | 256 | 8 | 1207.78 |  O2 |  Ascend_910A  | [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext) |
| edgenext_x_small | 75.14 | 256 | 8 | 1961.42 |  O3 |  Ascend_910A  | [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext) |
| edgenext_small | 79.15 | 256 | 8 | 882.00 |  O3 |  Ascend_910A  | [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext) |
| edgenext_base | 82.24 | 256 | 8 | 1151.98 |  O2 |  Ascend_910A  | [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext) |
| poolformer_s12 | 77.33 | 128 | 8 | 316.77 |  O3 |  Ascend_910A  | [mindcv_poolformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/poolformer) |
| xcit_tiny_12_p16 | 77.67 | 128 | 8 | 352.30 |  O2 |  Ascend_910A  | [mindcv_xcit](https://github.com/mindspore-lab/mindcv/blob/main/configs/xcit) |
| volo_d1 | 81.82 | 128 | 8 | 575.54 |  O3 |  Ascend_910A  | wait pr |
| swin_v2_t_w8_256 | 81.61 | 64 | 16 | 634.66 |  O0 |  Ascend_910A  | wait pr |
| cait | 82.25 | 64 | 8 | 435.54 |  O2 |  Ascend_910A  | waiting pr |


#### Object Detection
Accuracies are reported on COCO2017

| model | map@ | bs | cards | ms/step | amp | device | config | 
:-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| yolov7| 71.86 | 32 | 8 |  61.63  |  O2 |  Ascend_910A  | [mindyolo_yolov7](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg) | 








