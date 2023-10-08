## What Is New
- 2023.07.01: 🔥新增[语言模型](#语言模型) (包括热点模型glm/llama/bloom 来自于mindformers套件)
- 2023.06.01: 我们对经典SOTA模型进行了重构，模块化数据处理，模型定义，训练流程等常用组件，推出MindSpore CV/NLP/Audio/Yolo/OCR等系列
- 原models仓模型实现是基于MindSpore原生API，并且有一定训练推理加速优化
- 更多关于模型精度性能信息，请查阅 [benchmark](benchmark_CN.md)。

## 官方标准模型
 - [图像分类](#图像分类骨干类)
 - [目标检测](#目标检测)
 - [语义分割](#语义分割)
 - [OCR](#ocr)
 - [人脸](#人脸)
 - [语言模型](#语言模型)
### 计算机视觉

#### 图像分类（骨干类)

| model | acc@1 | mindcv recipe | vanilla mindspore |
| :-:     | :-:        | :-:    | :-:  |
|  vgg11           |  71.86   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   |
|  vgg13           |  72.87   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   |
|  vgg16           |  74.61   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/VGG/vgg16) |
|  vgg19           |  75.21   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/VGG/vgg19) |
| resnet18         |  70.21   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet)  |
| resnet34         |  74.15   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet)   |
| resnet50         |  76.69   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet)    |
| resnet101        |  78.24   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet)    |
| resnet152        |  78.72   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet)    |
| resnetv2_50      |  76.90   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnetv2)      |   |
| resnetv2_101     |  78.48   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnetv2)      |   |
| dpn92            |  79.46   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn)           |   |
| dpn98            |  79.94   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn)           |    |
| dpn107           |  80.05   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn)           |   |
| dpn131           |  80.07   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn)           |   |
| densenet121      |  75.64   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet)      |   |
| densenet161      |  79.09   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet)      |   |
| densenet169      |  77.26   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet)      |   |
| densenet201      |  78.14   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet)      |   |
| seresnet18       |  71.81   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |   |
| seresnet34       |  75.36   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |   |
| seresnet50       |  78.31   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |   |
| seresnext26      |  77.18   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |   |
| seresnext50      |  78.71   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |   |
| skresnet18       |  73.09   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/sknet)              |   |
| skresnet34       |  76.71   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/sknet)              |   |
| skresnet50_32x4d |  79.08   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/sknet)              |   |
| resnext50_32x4d  |  78.53   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext)            |   |
| resnext101_32x4d |  79.83   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext)            |   |
| resnext101_64x4d |  80.30   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext)            |   |
| resnext152_64x4d |  80.52   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext)            |   |
| rexnet_x09       |  77.07   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |   |
| rexnet_x10       |  77.38   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |   |
| rexnet_x13       |  79.06   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |   |
| rexnet_x15       |  79.94   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |   |
| rexnet_x20       |  80.64   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |   |
| resnest50        |  80.81   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnest)            |   |
| resnest101       |  82.50   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnest)            |   |
| res2net50        |  79.35   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net)            |   |
| res2net101       |  79.56   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net)            |   |
| res2net50_v1b    |  80.32   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net)            |   |
| res2net101_v1b   |  95.41   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net)            |   |
| googlenet        |  72.68   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/googlenet)          |   |
| inceptionv3      |  79.11   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/inceptionv3)        | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/inceptionv3) |
| inceptionv4      |  80.88   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/inceptionv4)        | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/inceptionv4) |
| mobilenet_v1_025 |  53.87   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |   |
| mobilenet_v1_050 |  65.94   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |   |
| mobilenet_v1_075 |  70.44   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |   |
| mobilenet_v1_100 |  72.95   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |   |
| mobilenet_v2_075 |  69.98   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2)        |   |
| mobilenet_v2_100 |  72.27   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2)        |   |
| mobilenet_v2_140 |  75.56   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2)        |   |
| mobilenet_v3_small     | 68.10 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv3)      |    |
| mobilenet_v3_large     | 75.23 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv3)      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv3) |
| shufflenet_v1_g3_x0_5  | 57.05 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv1)     |    |
| shufflenet_v1_g3_x1_5  | 67.77 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv1)     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv1) |
| shufflenet_v2_x0_5     | 57.05 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     |    |
| shufflenet_v2_x1_0     | 67.77 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv2) |
| shufflenet_v2_x1_5     | 57.05 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     |    |
| shufflenet_v2_x2_0     | 67.77 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     |    |
| xception               | 79.01 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/xception)         | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/xception) |
| ghostnet_50            | 66.03 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet)         |   |
| ghostnet_100           | 73.78 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet)         |   |
| ghostnet_130           | 75.50 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet)         |   |
| nasnet_a_4x1056        | 73.65 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/nasnet)           |   |
| mnasnet_0.5            | 68.07 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |   |
| mnasnet_0.75           | 71.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |   |
| mnasnet_1.0            | 74.28 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |   |
| mnasnet_1.4            | 76.01 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |   |
| efficientnet_b0        | 76.89 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/efficientnet)     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b0)
| efficientnet_b1        | 78.95 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/efficientnet)     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b1)
| efficientnet_b2        | 79.80 |     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b2) |
| efficientnet_b3        | 80.50 |     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b3) |
| efficientnet_v2        | 83.77 |     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnetv2) |
| regnet_x_200mf         | 68.74 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_x_400mf         | 73.16 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_x_600mf         | 73.34 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_x_800mf         | 76.04 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_y_200mf         | 70.30 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_y_400mf         | 73.91 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_y_600mf         | 75.69 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_y_800mf         | 76.52 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| mixnet_s               | 75.52 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mixnet)     |   |
| mixnet_m               | 76.64 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mixnet)     |   |
| mixnet_l               | 78.73 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mixnet)     |   |
| hrnet_w32              | 80.64 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/hrnet)      |   |
| hrnet_w48              | 81.19 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/hrnet)      |   |
| bit_resnet50           | 76.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/bit)         |   |
| bit_resnet50x3         | 80.63 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/bit)        |   |
| bit_resnet101          | 77.93 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/bit)        |   |
| repvgg_a0              | 72.19 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |   |
| repvgg_a1              | 74.19 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |   |
| repvgg_a2              | 76.63 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |   |
| repvgg_b0              | 74.99 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |   |
| repvgg_b1              | 78.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |   |
| repvgg_b2              | 79.29 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |   |
| repvgg_b3            | 80.46 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)       |   |
| repvgg_b1g2          | 78.03 |[config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)        |   |
| repvgg_b1g4          | 77.64 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)       |   |
| repvgg_b2g4          | 78.80 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)       |   |
| repmlp_t224          | 76.71 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repmlp)       |   |
| convnext_tiny        | 81.91 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convnext)     |   |
| convnext_small       | 83.40 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convnext)     |   |
| convnext_base        | 83.32 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convnext)     |   |
| vit_b_32_224         | 75.86 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit)          |  [link](https://gitee.com/mindspore/models/tree/master/official/cv/VIT) |
| vit_l_16_224         | 76.34| [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit)           |   |
| vit_l_32_224         | 73.71 |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit)         |   |
| swintransformer_tiny | 80.82 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/swintransformer) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SwinTransformer) |
| pvt_tiny            | 74.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt)           |   |
| pvt_small           | 79.66 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt)           |   |
| pvt_medium          | 81.82 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt)           |   |
| pvt_large           | 81.75 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt)           |   |
| pvt_v2_b0           | 71.50 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |   |
| pvt_v2_b1           | 78.91 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |   |
| pvt_v2_b2           | 81.99 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |   |
| pvt_v2_b3           | 82.84 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |   |
| pvt_v2_b4           | 83.14 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |   |
| pit_ti              | 72.96 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit)           |   |
| pit_xs              | 78.41 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit)           |   |
| pit_s               | 80.56 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit)           |   |
| pit_b               | 81.87 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit)           |   |
| coat_lite_tiny      | 77.35 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/coat)          |   |
| coat_lite_mini      | 78.51 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/coat)          |   |
| coat_tiny           | 79.67 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/coat)          |   |
| convit_tiny         | 73.66 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |   |
| convit_tiny_plus    | 77.00 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |   |
| convit_small        | 81.63 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |   |
| convit_small_plus   | 81.80 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |   |
| convit_base         | 82.10 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |   |
| convit_base_plus    | 81.96 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |   |
| crossvit_9          | 73.56 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/crossvit)      |   |
| crossvit_15         | 81.08 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/crossvit)      |   |
| crossvit_18         | 81.93 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/crossvit)      |   |
| mobilevit_xx_small  | 68.90 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilevit)     |   |
| mobilevit_x_small   | 74.98 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilevit)     |   |
| mobilevit_small     | 78.48 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilevit)     |   |
| visformer_tiny      | 78.28 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer)     |   |
| visformer_tiny_v2   | 78.82 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer)     |   |
| visformer_small     | 81.76 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer)     |   |
| visformer_small_v2  | 82.17 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer)     |   |
| edgenext_xx_small   | 71.02 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext)      |   |
| edgenext_x_small    | 75.14 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext)      |   |
| edgenext_small      | 79.15 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext)      |   |
| edgenext_base       | 82.24 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext)      |   |
| poolformer_s12      | 77.33 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/poolformer)    |   |
| xcit_tiny_12_p16    | 77.67 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/xcit)          |   |

### 目标检测

#### yolo

| model | map |  mindyolo recipe | vanilla mindspore |
|:-: | :-: | :-: | :-: |
| yolov8_n | 37.2 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |    |
| yolov8_s | 44.6 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |   |
| yolov8_m | 50.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |   |
| yolov8_l | 52.8 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |   |
| yolov8_x | 53.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |   |
| yolov7_t | 37.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |    |
| yolov7_l | 50.8 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |   |
| yolov7_x |  52.4| [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |   |
| yolov5_n | 27.3 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |   |
| yolov5_s | 37.6 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv5) |
| yolov5_m | 44.9 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |   |
| yolov5_l | 48.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |   |
| yolov5_x | 50.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |   |
| yolov4_csp       | 45.4 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov4) |   |
| yolov4_csp(silu) | 45.8 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov4) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv4) |
| yolov3_darknet53 | 45.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov3) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv3) |
| yolox_n | 24.1 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_t | 33.3 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_s | 40.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_m | 46.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_l | 49.2 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_x | 51.6 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_darknet53 | 47.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |

#### 经典

| model |  map | mind_series recipe | vanilla mindspore |
| :-:     |  :-:            | :-:        |  :-:  |
|  ssd_vgg16                 | 23.2  |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SSD)|
|  ssd_mobilenetv1           | 22.0  |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SSD)|
|  ssd_mobilenetv2           | 29.1  |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SSD)|
|  ssd_resnet50              | 34.3  |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SSD)|
|  fasterrcnn                  | 58    | [link](https://github.com/mindspore-lab/models/tree/master/official/cv/RCNN))  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |
|  maskrcnn_mobilenetv1      | coming soon   |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/MaskRCNN/maskrcnn_mobilenetv1) |
|  maskrcnn_resnet50         | coming soon   |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/MaskRCNN/maskrcnn_resnet50) |

### 语义分割

| model |  mind_series recipe | vanilla mindspore |
| :-:     |  :-:            | :-:     |
| ocrnet          |   [link](https://github.com/mindspore-lab/models/tree/master/official/cv/OCRNet)   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/OCRNet/)         |
| deeplab v3      |      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabv3)       |
| deeplab v3 plus |      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabV3P)      |
| unet            |      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Unet)            |
| unet3d          |      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Unet3d)          |

### OCR



#### 文本检测

| model  |dataset | fscore | mindocr recipe | vanilla mindspore |
:-:     |   :-:       | :-:        | :-:   |  :-:   |
| dbnet_mobilenetv3  | icdar2015          | 77.23 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DBNet/)  |
| dbnet_resnet18     | icdar2015          | 81.73 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DBNet/)  |
| dbnet_resnet50     | icdar2015          | 85.05 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DBNet/)  |
| dbnet++_resnet50   | icdar2015          | 86.74 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |   |
| psenet_resnet152   | icdar2015          | 82.06 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) | [link](https://gitee.com/mindspore/models/tree/master/research/cv/psenet)  |
| east_resnet50      | icdar2015          | 84.87 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east)   | [link](https://gitee.com/mindspore/models/tree/master/research/cv/east)    |
| fcenet_resnet50    | icdar2015          | 84.12 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/fcenet)   |   |

#### 文本识别

| model | dataset | acc | mindocr recipe | vanilla mindspore |
:-:     |   :-:       | :-:        | :-:   |  :-:   |
| svtr_tiny          | IC03,13,15,IIIT,etc | 89.02 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/svtr)   |   |
| crnn_vgg7          | IC03,13,15,IIIT,etc | 82.03 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/CRNN)    |
| crnn_resnet34_vd   | IC03,13,15,IIIT,etc | 84.45 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |   |
| rare_resnet34_vd   | IC03,13,15,IIIT,etc | 85.19 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare)   | [link](https://gitee.com/mindspore/models/tree/master/research/cv/crnn_seq2seq_ocr)  |

#### 文本方向分类

| model | dataset | acc | mindocr recipe |
:-:     |   :-:       | :-:        | :-:   |
| mobilenetv3  | RCTW17,MTWI,LSVT | 94.59 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/cls/mobilenetv3)   |

### 人脸

| model | dataset | acc | mindface recipe | vanilla mindspore
| :-:     |  :-:       | :-:        | :-:   | :-: |
| arcface_mobilefacenet-0.45g  | MS1MV2          | 98.70  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |   |
| arcface_r50                  | MS1MV2          | 99.76  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |    |
| arcface_r100                 | MS1MV2          | 99.38  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Arcface) |
| arcface_vit_t                | MS1MV2          | 99.71  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |   |
| arcface_vit_s                | MS1MV2          | 99.76  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |    |
| arcface_vit_b                | MS1MV2          | 99.81  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |    |
| arcface_vit_l                | MS1MV2          | 99.75  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |    |
| retinaface_mobilenet_0.25    | WiderFace        | 90.77/88.2/74.76  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/detection)  | [link](https://gitee.com/mindspore/models/tree/master/research/cv/retinaface) |
| retinaface_r50               | WiderFace        | 95.07/93.61/84.84 | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/detection)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/RetinaFace_ResNet50) |

### 语言模型
| model |  mindformer recipe | vanilla mindspore
| :-:     |  :-:   | :-: |
| bert_base   | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/t5.md) | [link](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert) |
| t5_small    | [config](https://github.com/mindspore-lab/mindformers/blob/master/docs/model_cards/bert.md) |  |
| gpt2_small  | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/gpt2.md) |
| gpt2_13b    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/gpt2.md) |
| gpt2_52b    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/gpt2.md) |
| pangu_alpha | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/pangualpha.md) | 
| glm_6b       | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/glm.md)  |
| glm_6b_lora  | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/glm.md)  |
| llama_7b     | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md) |
| llama_13b    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md) |
| llama_65b    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md) |
| llama_7b_lora | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md) |
| bloom_560m    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bloom.md) |
| bloom_7.1b    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bloom.md) |
| bloom_65b     | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bloom.md) |
| bloom_176b    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bloom.md) |

## 免责声明

MindSpore仅提供下载和预处理公共数据集的脚本。我们不拥有这些数据集，也不对它们的质量负责或维护。请确保您具有在数据集许可下使用该数据集的权限。在这些数据集上训练的模型仅用于非商业研究和教学目的。

致数据集拥有者：如果您不希望将数据集包含在MindSpore中，或者希望以任何方式对其进行更新，我们将根据要求删除或更新所有公共内容。请通过GitHub或Gitee与我们联系。非常感谢您对这个社区的理解和贡献。

## 许可证

[Apache 2.0许可证](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)
