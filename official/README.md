## What Is New

- 2023.07.01: 🔥add [llm](#nlp) model (including hot glm/llama/bloom from mindformers)
- 2023.06.01: We've done code refactoring for classic SOTA models,modularized data processing, model definition&creation, training process and other common components with new lanched MindSpore CV/NLP/Audio/Yolo/OCR Series toolbox

- Old models were implemented by original MindSpore API with some tricks for training process speedup

- More information for model performance, please check [benchmark](benchmark.md).

## Standard Models
 - [image classification](#image-classification-backbone)
 - [object detection](#object-detection)
 - [semantic segmentation](#semantic-segmentation)
 - [ocr](#ocr)
 - [face](#face)
 - [llm](#nlp)

### Computer Vision

#### Image Classification (backbone)

| model                  | acc@1 | mindcv recipe                                                | vanilla mindspore                                            |
| ---------------------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| bit_resnet101          | 77.93 | [mindcv_bit](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit/bit_resnet101_ascend.yaml) |                                                              |
| bit_resnet50           | 76.81 | [mindcv_bit](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit/bit_resnet50_ascend.yaml) |                                                              |
| bit_resnet50x3         | 80.63 | [mindcv_bit](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit/bit_resnet50x3_ascend.yaml) |                                                              |
| cait_s24               | 82.25 | uploading                                                    |                                                              |
| cmt_small              | 83.24 | [mindcv_cmt](https://github.com/mindspore-lab/mindcv/blob/main/configs/cmt/cmt_small_ascend.yaml) |                                                              |
| coat_lite_mini         | 78.51 | [mindcv_coat](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat/coat_lite_mini_ascend.yaml) |                                                              |
| coat_lite_tiny         | 77.35 | [mindcv_coat](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat/coat_lite_tiny_ascend.yaml) |                                                              |
| coat_mini              | 81.08 | [mindcv_coat](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat/coat_mini_ascend.yaml) |                                                              |
| coat_tiny              | 79.67 | [mindcv_coat](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat/coat_tiny_ascend.yaml) |                                                              |
| convit_base            | 82.1  | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_base_ascend.yaml) |                                                              |
| convit_base_plus       | 81.96 | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_base_plus_ascend.yaml) |                                                              |
| convit_small           | 81.63 | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_small_ascend.yaml) |                                                              |
| convit_small_plus      | 81.8  | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_small_plus_ascend.yaml) |                                                              |
| convit_tiny            | 73.66 | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_tiny_ascend.yaml) |                                                              |
| convit_tiny_plus       | 77    | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_tiny_plus_ascend.yaml) |                                                              |
| convnext_base          | 83.32 | [mindcv_convnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnext/convnext_base_ascend.yaml) |                                                              |
| convnext_small         | 83.4  | [mindcv_convnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnext/convnext_small_ascend.yaml) |                                                              |
| convnext_tiny          | 81.91 | [mindcv_convnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnext/convnext_tiny_ascend.yaml) |                                                              |
| convnextv2_tiny        | 82.43 | [mindcv_convnextv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnextv2/convnextv2_tiny_ascend.yaml) |                                                              |
| crossvit_15            | 81.08 | [mindcv_crossvit](https://github.com/mindspore-lab/mindcv/blob/main/configs/crossvit/crossvit_15_ascend.yaml) |                                                              |
| crossvit_18            | 81.93 | [mindcv_crossvit](https://github.com/mindspore-lab/mindcv/blob/main/configs/crossvit/crossvit_18_ascend.yaml) |                                                              |
| crossvit_9             | 73.56 | [mindcv_crossvit](https://github.com/mindspore-lab/mindcv/blob/main/configs/crossvit/crossvit_9_ascend.yaml) |                                                              |
| densenet121            | 75.64 | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_121_ascend.yaml) |                                                              |
| densenet161            | 79.09 | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_161_ascend.yaml) |                                                              |
| densenet169            | 77.26 | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_169_ascend.yaml) |                                                              |
| densenet201            | 78.14 | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_201_ascend.yaml) |                                                              |
| dpn107                 | 80.05 | [mindcv_dpn](https://github.com/mindspore-lab/mindcv/blob/main/configs/dpn/dpn107_ascend.yaml) |                                                              |
| dpn131                 | 80.07 | [mindcv_dpn](https://github.com/mindspore-lab/mindcv/blob/main/configs/dpn/dpn131_ascend.yaml) |                                                              |
| dpn92                  | 79.46 | [mindcv_dpn](https://github.com/mindspore-lab/mindcv/blob/main/configs/dpn/dpn92_ascend.yaml) |                                                              |
| dpn98                  | 79.94 | [mindcv_dpn](https://github.com/mindspore-lab/mindcv/blob/main/configs/dpn/dpn98_ascend.yaml) |                                                              |
| edgenext_base          | 82.24 | [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext_base_ascend.yaml) |                                                              |
| edgenext_small         | 79.15 | [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext_small_ascend.yaml) |                                                              |
| edgenext_x_small       | 75.14 | [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext_x_small_ascend.yaml) |                                                              |
| edgenext_xx_small      | 71.02 | [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext_xx_small_ascend.yaml) |                                                              |
| efficientnet_b0        | 76.89 | [mindcv_efficientnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/efficientnet/efficientnet_b0_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b0) |
| efficientnet_b1        | 78.95 | [mindcv_efficientnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/efficientnet/efficientnet_b1_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b1) |
| efficientnet_b2        | 79.80 |                                                              | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b2) |
| efficientnet_b3        | 80.50 |                                                              | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b3) |
| efficientnet_v2        | 83.77 |                                                              | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnetv2) |
| ghostnet_050           | 66.03 | [mindcv_ghostnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/ghostnet/ghostnet_050_ascend.yaml) |                                                              |
| ghostnet_100           | 73.78 | [mindcv_ghostnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/ghostnet/ghostnet_100_ascend.yaml) |                                                              |
| ghostnet_130           | 75.5  | [mindcv_ghostnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/ghostnet/ghostnet_130_ascend.yaml) |                                                              |
| googlenet              | 72.68 | [mindcv_googlenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/googlenet/googlenet_ascend.yaml) |                                                              |
| halonet_50t            | 79.53 | [mindcv_halonet](https://github.com/mindspore-lab/mindcv/blob/main/configs/halonet/halonet_50t_ascend.yaml) |                                                              |
| hrnet_w32              | 80.64 | [mindcv_hrnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/hrnet/hrnet_w32_ascend.yaml) |                                                              |
| hrnet_w48              | 81.19 | [mindcv_hrnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/hrnet/hrnet_w48_ascend.yaml) |                                                              |
| inception_v3           | 79.11 | [mindcv_inception](https://github.com/mindspore-lab/mindcv/blob/main/configs/inceptionv3/inception_v3_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/inceptionv3) |
| inception_v4           | 80.88 | [mindcv_inception](https://github.com/mindspore-lab/mindcv/blob/main/configs/inceptionv4/inception_v4_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/inceptionv4) |
| mixnet_l               | 78.73 | [mindcv_mixnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mixnet/mixnet_l_ascend.yaml) |                                                              |
| mixnet_m               | 76.64 | [mindcv_mixnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mixnet/mixnet_m_ascend.yaml) |                                                              |
| mixnet_s               | 75.52 | [mindcv_mixnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mixnet/mixnet_s_ascend.yaml) |                                                              |
| mnasnet_050            | 68.07 | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_0.5_ascend.yaml) |                                                              |
| mnasnet_075            | 71.81 | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_0.75_ascend.yaml) |                                                              |
| mnasnet_100            | 74.28 | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_1.0_ascend.yaml) |                                                              |
| mnasnet_130            | 75.65 | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_1.3_ascend.yaml) |                                                              |
| mnasnet_140            | 76.01 | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_1.4_ascend.yaml) |                                                              |
| mobilenet_v1_025       | 53.87 | [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1/mobilenet_v1_0.25_ascend.yaml) |                                                              |
| mobilenet_v1_050       | 65.94 | [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1/mobilenet_v1_0.5_ascend.yaml) |                                                              |
| mobilenet_v1_075       | 70.44 | [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1/mobilenet_v1_0.75_ascend.yaml) |                                                              |
| mobilenet_v1_100       | 72.95 | [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1/mobilenet_v1_1.0_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv1) |
| mobilenet_v2_075       | 69.98 | [mindcv_mobilenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2/mobilenet_v2_0.75_ascend.yaml) |                                                              |
| mobilenet_v2_100       | 72.27 | [mindcv_mobilenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2/mobilenet_v2_1.0_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv2) |
| mobilenet_v2_140       | 75.56 | [mindcv_mobilenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2/mobilenet_v2_1.4_ascend.yaml) |                                                              |
| mobilenet_v3_large_100 | 75.23 | [mindcv_mobilenetv3](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv3/mobilenet_v3_large_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv3) |
| mobilenet_v3_small_100 | 68.1  | [mindcv_mobilenetv3](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv3/mobilenet_v3_small_ascend.yaml) |                                                              |
| mobilevit_small        | 78.47 | [mindcv_mobilevit](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilevit/mobilevit_small_ascend.yaml) |                                                              |
| mobilevit_x_small      | 74.99 | [mindcv_mobilevit](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilevit/mobilevit_x_small_ascend.yaml) |                                                              |
| mobilevit_xx_small     | 68.91 | [mindcv_mobilevit](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilevit/mobilevit_xx_small_ascend.yaml) |                                                              |
| nasnet_a_4x1056        | 73.65 | [mindcv_nasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/nasnet/nasnet_a_4x1056_ascend.yaml) |                                                              |
| pit_b                  | 81.87 | [mindcv_pit](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit/pit_b_ascend.yaml) |                                                              |
| pit_s                  | 80.56 | [mindcv_pit](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit/pit_s_ascend.yaml) |                                                              |
| pit_ti                 | 72.96 | [mindcv_pit](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit/pit_ti_ascend.yaml) |                                                              |
| pit_xs                 | 78.41 | [mindcv_pit](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit/pit_xs_ascend.yaml) |                                                              |
| poolformer_s12         | 77.33 | [mindcv_poolformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/poolformer/poolformer_s12_ascend.yaml) |                                                              |
| pvt_large              | 81.75 | [mindcv_pvt](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_large_ascend.yaml) |                                                              |
| pvt_medium             | 81.82 | [mindcv_pvt](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_medium_ascend.yaml) |                                                              |
| pvt_small              | 79.66 | [mindcv_pvt](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_small_ascend.yaml) |                                                              |
| pvt_tiny               | 74.81 | [mindcv_pvt](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_tiny_ascend.yaml) |                                                              |
| pvt_v2_b0              | 71.5  | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2/pvt_v2_b0_ascend.yaml) |                                                              |
| pvt_v2_b1              | 78.91 | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2/pvt_v2_b1_ascend.yaml) |                                                              |
| pvt_v2_b2              | 81.99 | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2/pvt_v2_b2_ascend.yaml) |                                                              |
| pvt_v2_b3              | 82.84 | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2/pvt_v2_b3_ascend.yaml) |                                                              |
| pvt_v2_b4              | 83.14 | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2/pvt_v2_b4_ascend.yaml) |                                                              |
| regnet_x_200mf         | 68.74 | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_200mf_ascend.yaml) |                                                              |
| regnet_x_400mf         | 73.16 | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_400mf_ascend.yaml) |                                                              |
| regnet_x_600mf         | 74.34 | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_600mf_ascend.yaml) |                                                              |
| regnet_x_800mf         | 76.04 | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_800mf_ascend.yaml) |                                                              |
| regnet_y_16gf          | 82.92 | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_16gf_ascend.yaml) |                                                              |
| regnet_y_200mf         | 70.3  | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_200mf_ascend.yaml) |                                                              |
| regnet_y_400mf         | 73.91 | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_400mf_ascend.yaml) |                                                              |
| regnet_y_600mf         | 75.69 | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_600mf_ascend.yaml) |                                                              |
| regnet_y_800mf         | 76.52 | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_800mf_ascend.yaml) |                                                              |
| repmlp_t224            | 76.71 | [mindcv_repmlp](https://github.com/mindspore-lab/mindcv/blob/main/configs/repmlp/repmlp_t224_ascend.yaml) |                                                              |
| repvgg_a0              | 72.19 | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_a0_ascend.yaml) |                                                              |
| repvgg_a1              | 74.19 | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_a1_ascend.yaml) |                                                              |
| repvgg_a2              | 76.63 | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_a2_ascend.yaml) |                                                              |
| repvgg_b0              | 74.99 | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b0_ascend.yaml) |                                                              |
| repvgg_b1              | 78.81 | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b1_ascend.yaml) |                                                              |
| repvgg_b1g2            | 78.03 | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b1g2_ascend.yaml) |                                                              |
| repvgg_b1g4            | 77.64 | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b1g4_ascend.yaml) |                                                              |
| repvgg_b2              | 79.29 | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b2_ascend.yaml) |                                                              |
| repvgg_b2g4            | 78.8  | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b2g4_ascend.yaml) |                                                              |
| repvgg_b3              | 80.46 | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b3_ascend.yaml) |                                                              |
| res2net101             | 79.56 | [mindcv_res2net](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_101_ascend.yaml) |                                                              |
| res2net101_v1b         | 81.14 | [mindcv_res2net](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_101_v1b_ascend.yaml) |                                                              |
| res2net50              | 79.35 | [mindcv_res2net](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_50_ascend.yaml) |                                                              |
| res2net50_v1b          | 80.32 | [mindcv_res2net](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_50_v1b_ascend.yaml) |                                                              |
| resnest101             | 82.9  | [mindcv_resnest](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnest/resnest101_ascend.yaml) |                                                              |
| resnest50              | 80.81 | [mindcv_resnest](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnest/resnest50_ascend.yaml) |                                                              |
| resnet101              | 78.24 | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_101_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |
| resnet152              | 78.72 | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_152_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |
| resnet18               | 70.21 | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_18_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |
| resnet34               | 74.15 | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_34_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |
| resnet50               | 76.69 | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_50_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |
| resnetv2_101           | 78.48 | [mindcv_resnetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnetv2/resnetv2_101_ascend.yaml) |                                                              |
| resnetv2_50            | 76.9  | [mindcv_resnetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnetv2/resnetv2_50_ascend.yaml) |                                                              |
| resnext101_32x4d       | 79.83 | [mindcv_resnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnext/resnext101_32x4d_ascend.yaml) |                                                              |
| resnext101_64x4d       | 80.3  | [mindcv_resnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnext/resnext101_64x4d_ascend.yaml) |                                                              |
| resnext152_64x4d       | 80.52 | [mindcv_resnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnext/resnext152_64x4d_ascend.yaml) |                                                              |
| resnext50_32x4d        | 78.53 | [mindcv_resnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnext/resnext50_32x4d_ascend.yaml) |                                                              |
| rexnet_09              | 77.06 | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet/rexnet_x09_ascend.yaml) |                                                              |
| rexnet_10              | 77.38 | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet/rexnet_x10_ascend.yaml) |                                                              |
| rexnet_13              | 79.06 | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet/rexnet_x13_ascend.yaml) |                                                              |
| rexnet_15              | 79.95 | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet/rexnet_x15_ascend.yaml) |                                                              |
| rexnet_20              | 80.64 | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet/rexnet_x20_ascend.yaml) |                                                              |
| seresnet18             | 71.81 | [mindcv_seresnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnet18_ascend.yaml) |                                                              |
| seresnet34             | 75.38 | [mindcv_seresnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnet34_ascend.yaml) |                                                              |
| seresnet50             | 78.32 | [mindcv_seresnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnet50_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |
| seresnext26_32x4d      | 77.17 | [mindcv_seresnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnext26_32x4d_ascend.yaml) |                                                              |
| seresnext50_32x4d      | 78.71 | [mindcv_seresnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnext50_32x4d_ascend.yaml) |                                                              |
| shufflenet_v1_g3_05    | 57.05 | [mindcv_shufflenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv1/shufflenet_v1_0.5_ascend.yaml) |                                                              |
| shufflenet_v1_g3_10    | 67.77 | [mindcv_shufflenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv1/shufflenet_v1_1.0_ascend.yaml) |                                                              |
| shufflenet_v2_x0_5     | 60.53 | [mindcv_shufflenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv2/shufflenet_v2_0.5_ascend.yaml) |                                                              |
| shufflenet_v2_x1_0     | 69.47 | [mindcv_shufflenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv2/shufflenet_v2_1.0_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv2) |
| shufflenet_v2_x1_5     | 72.79 | [mindcv_shufflenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv2/shufflenet_v2_1.5_ascend.yaml) |                                                              |
| shufflenet_v2_x2_0     | 75.07 | [mindcv_shufflenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv2/shufflenet_v2_2.0_ascend.yaml) |                                                              |
| skresnet18             | 73.09 | [mindcv_skresnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/sknet/skresnet18_ascend.yaml) |                                                              |
| skresnet34             | 76.71 | [mindcv_skresnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/sknet/skresnet34_ascend.yaml) |                                                              |
| skresnext50_32x4d      | 79.08 | [mindcv_skresnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/sknet/skresnext50_32x4d_ascend.yaml) |                                                              |
| squeezenet1_0          | 59.01 | [mindcv_squeezenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/squeezenet/squeezenet_1.0_ascend.yaml) |                                                              |
| squeezenet1_1          | 58.44 | [mindcv_squeezenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/squeezenet/squeezenet_1.1_ascend.yaml) |                                                              |
| swin_tiny              | 80.82 | [mindcv_swin](https://github.com/mindspore-lab/mindcv/blob/main/configs/swintransformer/swin_tiny_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SwinTransformer) |
| swinv2_tiny_window8    | 81.42 | [mindcv_swinv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/swintransformerv2/swinv2_tiny_window8_ascend.yaml) |                                                              |
| vgg11                  | 71.86 | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg/vgg11_ascend.yaml) |                                                              |
| vgg13                  | 72.87 | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg/vgg13_ascend.yaml) |                                                              |
| vgg16                  | 74.61 | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg/vgg16_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/VGG/vgg16) |
| vgg19                  | 75.21 | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg/vgg19_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/VGG/vgg19) |
| visformer_small        | 81.76 | [mindcv_visformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer/visformer_small_ascend.yaml) |                                                              |
| visformer_small_v2     | 82.17 | [mindcv_visformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer/visformer_small_v2_ascend.yaml) |                                                              |
| visformer_tiny         | 78.28 | [mindcv_visformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer/visformer_tiny_ascend.yaml) |                                                              |
| visformer_tiny_v2      | 78.82 | [mindcv_visformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer/visformer_tiny_v2_ascend.yaml) |                                                              |
| vit_b_32_224           | 75.86 | [mindcv_vit](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit/vit_b32_224_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/VIT) |
| vit_l_16_224           | 76.34 | [mindcv_vit](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit/vit_l16_224_ascend.yaml) |                                                              |
| vit_l_32_224           | 73.71 | [mindcv_vit](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit/vit_l32_224_ascend.yaml) |                                                              |
| volo_d1                | 82.59 | [mindcv_volo](https://github.com/mindspore-lab/mindcv/blob/main/configs/volo/volo_d1_ascend.yaml) |                                                              |
| xception               | 79.01 | [mindcv_xception](https://github.com/mindspore-lab/mindcv/blob/main/configs/xception/xception_ascend.yaml) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/xception) |
| xcit_tiny_12_p16_224   | 77.67 | [mindcv_xcit](https://github.com/mindspore-lab/mindcv/blob/main/configs/xcit/xcit_tiny_12_p16_ascend.yaml) |                                                              |

### Object Detection

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

#### Classic

| model |  map | mind_series recipe | vanilla mindspore |
| :-:     |  :-:            | :-:        |  :-:  |
|  ssd_vgg16                 | 23.2  |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SSD)|
|  ssd_mobilenetv1           | 22.0  |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SSD)|
|  ssd_mobilenetv2           | 29.1  |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SSD)|
|  ssd_resnet50              | 34.3  |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SSD)|
|  fasterrcnn                  | 58    | [link](https://github.com/mindspore-lab/models/tree/master/official/cv/RCNN))  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |
|  maskrcnn_mobilenetv1      | coming soon   |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/MaskRCNN/maskrcnn_mobilenetv1) |
|  maskrcnn_resnet50         | coming soon   |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/MaskRCNN/maskrcnn_resnet50) |

### Semantic Segmentation

| model |  mind_series recipe | vanilla mindspore |
| :-:     |  :-:            | :-:     |
| ocrnet          |   [link](https://github.com/mindspore-lab/models/tree/master/official/cv/OCRNet)   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/OCRNet/)         |
| deeplab v3      |      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabv3)       |
| deeplab v3 plus |      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabV3P)      |
| unet            |      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Unet)            |
| unet3d          |      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Unet3d)          |

### OCR


### Text Detection

| model  |dataset | F-score | mindocr recipe | vanilla mindspore |
:-:     |   :-:       | :-:        | :-:   |  :-:   |
| dbnet_mobilenetv3  | icdar2015          | 77.23 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DBNet/)  |
| dbnet_resnet18     | icdar2015          | 81.73 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DBNet/)  |
| dbnet_resnet50     | icdar2015          | 85.05 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DBNet/)  |
| dbnet++_resnet50   | icdar2015          | 86.74 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |   |
| psenet_resnet152   | icdar2015          | 82.06 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) | [link](https://gitee.com/mindspore/models/tree/master/research/cv/psenet)  |
| east_resnet50      | icdar2015          | 84.87 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east)   | [link](https://gitee.com/mindspore/models/tree/master/research/cv/east)    |
| fcenet_resnet50    | icdar2015          | 84.12 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/fcenet)   |   |

### Text Recognition

| model | dataset | acc | mindocr recipe | vanilla mindspore |
:-:     |   :-:       | :-:        | :-:   |  :-:   |
| svtr_tiny          | IC03,13,15,IIIT,etc | 89.02 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/svtr)   |   |
| crnn_vgg7          | IC03,13,15,IIIT,etc | 82.03 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/CRNN)    |
| crnn_resnet34_vd   | IC03,13,15,IIIT,etc | 84.45 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |   |
| rare_resnet34_vd   | IC03,13,15,IIIT,etc | 85.19 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare)   | [link](https://gitee.com/mindspore/models/tree/master/research/cv/crnn_seq2seq_ocr)  |

### Text Direction Classification

| model | dataset | acc | mindocr recipe | vanilla mindspore |
:-:     |   :-:       | :-:        | :-:   | :-: |
| mobilenetv3    | RCTW17,MTWI,LSVT | 94.59 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/cls/mobilenetv3)   |


### Face

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


### nlp
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

### audio coming soon


## Disclaimers

Mindspore only provides scripts that downloads and preprocesses public datasets. We do not own these datasets and are not responsible for their quality or maintenance. Please make sure you have permission to use the dataset under the dataset’s license. The models trained on these dataset are for non-commercial research and educational purpose only.

To dataset owners: we will remove or update all public content upon request if you don’t want your dataset included on Mindspore, or wish to update it in any way. Please contact us through a Github/Gitee issue. Your understanding and contribution to this community is greatly appreciated.

MindSpore is Apache 2.0 licensed. Please see the LICENSE file.

## License

[Apache License 2.0](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)
