### Computer Vision

##### Currently all results are tested on Ascend 910, RTX 3090 version is coming soon

### Image Classification

#### Accuracies are reported on ImageNet-1K

| model                  | acc@1 | bs   | cards | ms/step   | amp  | config                                                       |
| ---------------------- | ----- | ---- | ----- | --------- | ---- | ------------------------------------------------------------ |
| bit_resnet101          | 77.93 | 16   | 8     | 128.15    | O0   | [mindcv_bit](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit/bit_resnet101_ascend.yaml) |
| bit_resnet50           | 76.81 | 32   | 8     | 130.6     | O0   | [mindcv_bit](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit/bit_resnet50_ascend.yaml) |
| bit_resnet50x3         | 80.63 | 32   | 8     | 533.09    | O0   | [mindcv_bit](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit/bit_resnet50x3_ascend.yaml) |
| cait_s24               | 82.25 | 64   | 8     | 435.54    | O2   | uploading                                                    |
| cmt_small              | 83.24 | 128  | 8     | uploading | O2   | [mindcv_cmt](https://github.com/mindspore-lab/mindcv/blob/main/configs/cmt/cmt_small_ascend.yaml) |
| coat_lite_mini         | 78.51 | 64   | 8     | 265.44    | O2   | [mindcv_coat](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat/coat_lite_mini_ascend.yaml) |
| coat_lite_tiny         | 77.35 | 64   | 8     | 258.07    | O2   | [mindcv_coat](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat/coat_lite_tiny_ascend.yaml) |
| coat_mini              | 81.08 | 32   | 8     | uploading | O2   | [mindcv_coat](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat/coat_mini_ascend.yaml) |
| coat_tiny              | 79.67 | 64   | 8     | 580.54    | O2   | [mindcv_coat](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat/coat_tiny_ascend.yaml) |
| convit_base            | 82.1  | 128  | 8     | 701.84    | O2   | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_base_ascend.yaml) |
| convit_base_plus       | 81.96 | 128  | 8     | 983.21    | O2   | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_base_plus_ascend.yaml) |
| convit_small           | 81.63 | 192  | 8     | 588.73    | O2   | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_small_ascend.yaml) |
| convit_small_plus      | 81.8  | 192  | 8     | 665.74    | O2   | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_small_plus_ascend.yaml) |
| convit_tiny            | 73.66 | 256  | 8     | 388.8     | O2   | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_tiny_ascend.yaml) |
| convit_tiny_plus       | 77    | 256  | 8     | 393.6     | O2   | [mindcv_convit](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_tiny_plus_ascend.yaml) |
| convnext_base          | 83.32 | 128  | 8     | 531.1     | O2   | [mindcv_convnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnext/convnext_base_ascend.yaml) |
| convnext_small         | 83.4  | 128  | 8     | 405.96    | O2   | [mindcv_convnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnext/convnext_small_ascend.yaml) |
| convnext_tiny          | 81.91 | 128  | 8     | 343.21    | O2   | [mindcv_convnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnext/convnext_tiny_ascend.yaml) |
| convnextv2_tiny        | 82.43 | 128  | 8     | uploading | O2   | [mindcv_convnextv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnextv2/convnextv2_tiny_ascend.yaml) |
| crossvit_15            | 81.08 | 256  | 8     | 1086      | O3   | [mindcv_crossvit](https://github.com/mindspore-lab/mindcv/blob/main/configs/crossvit/crossvit_15_ascend.yaml) |
| crossvit_18            | 81.93 | 256  | 8     | 1137.6    | O3   | [mindcv_crossvit](https://github.com/mindspore-lab/mindcv/blob/main/configs/crossvit/crossvit_18_ascend.yaml) |
| crossvit_9             | 73.56 | 256  | 8     | 685.25    | O3   | [mindcv_crossvit](https://github.com/mindspore-lab/mindcv/blob/main/configs/crossvit/crossvit_9_ascend.yaml) |
| densenet121            | 75.64 | 32   | 8     | 48.07     | O2   | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_121_ascend.yaml) |
| densenet161            | 79.09 | 32   | 8     | 115.11    | O2   | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_161_ascend.yaml) |
| densenet169            | 77.26 | 32   | 8     | 73.14     | O2   | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_169_ascend.yaml) |
| densenet201            | 78.14 | 32   | 8     | 96.12     | O2   | [mindcv_densenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_201_ascend.yaml) |
| dpn107                 | 80.05 | 32   | 8     | 107.6     | O2   | [mindcv_dpn](https://github.com/mindspore-lab/mindcv/blob/main/configs/dpn/dpn107_ascend.yaml) |
| dpn131                 | 80.07 | 32   | 8     | 143.57    | O2   | [mindcv_dpn](https://github.com/mindspore-lab/mindcv/blob/main/configs/dpn/dpn131_ascend.yaml) |
| dpn92                  | 79.46 | 32   | 8     | 79.89     | O2   | [mindcv_dpn](https://github.com/mindspore-lab/mindcv/blob/main/configs/dpn/dpn92_ascend.yaml) |
| dpn98                  | 79.94 | 32   | 8     | 106.6     | O2   | [mindcv_dpn](https://github.com/mindspore-lab/mindcv/blob/main/configs/dpn/dpn98_ascend.yaml) |
| edgenext_base          | 82.24 | 256  | 8     | 1151.98   | O2   | [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext_base_ascend.yaml) |
| edgenext_small         | 79.15 | 256  | 8     | 882       | O3   | [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext_small_ascend.yaml) |
| edgenext_x_small       | 75.14 | 256  | 8     | 1961.42   | O3   | [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext_x_small_ascend.yaml) |
| edgenext_xx_small      | 71.02 | 256  | 8     | 1207.78   | O2   | [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext_xx_small_ascend.yaml) |
| efficientnet_b0        | 76.89 | 128  | 8     | 276.77    | O2   | [mindcv_efficientnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/efficientnet/efficientnet_b0_ascend.yaml) |
| efficientnet_b1        | 78.95 | 128  | 8     | 435.9     | O2   | [mindcv_efficientnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/efficientnet/efficientnet_b1_ascend.yaml) |
| ghostnet_050           | 66.03 | 128  | 8     | 220.88    | O3   | [mindcv_ghostnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/ghostnet/ghostnet_050_ascend.yaml) |
| ghostnet_100           | 73.78 | 128  | 8     | 222.67    | O3   | [mindcv_ghostnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/ghostnet/ghostnet_100_ascend.yaml) |
| ghostnet_130           | 75.5  | 128  | 8     | 223.11    | O3   | [mindcv_ghostnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/ghostnet/ghostnet_130_ascend.yaml) |
| googlenet              | 72.68 | 32   | 8     | 23.26     | O0   | [mindcv_googlenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/googlenet/googlenet_ascend.yaml) |
| halonet_50t            | 79.53 | 64   | 8     | uploading | O2   | [mindcv_halonet](https://github.com/mindspore-lab/mindcv/blob/main/configs/halonet/halonet_50t_ascend.yaml) |
| hrnet_w32              | 80.64 | 128  | 8     | 335.73    | O2   | [mindcv_hrnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/hrnet/hrnet_w32_ascend.yaml) |
| hrnet_w48              | 81.19 | 128  | 8     | 463.63    | O2   | [mindcv_hrnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/hrnet/hrnet_w48_ascend.yaml) |
| inception_v3           | 79.11 | 32   | 8     | 49.96     | O0   | [mindcv_inception](https://github.com/mindspore-lab/mindcv/blob/main/configs/inceptionv3/inception_v3_ascend.yaml) |
| inception_v4           | 80.88 | 32   | 8     | 93.33     | O0   | [mindcv_inception](https://github.com/mindspore-lab/mindcv/blob/main/configs/inceptionv4/inception_v4_ascend.yaml) |
| mixnet_l               | 78.73 | 128  | 8     | 389.97    | O3   | [mindcv_mixnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mixnet/mixnet_l_ascend.yaml) |
| mixnet_m               | 76.64 | 128  | 8     | 384.68    | O3   | [mindcv_mixnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mixnet/mixnet_m_ascend.yaml) |
| mixnet_s               | 75.52 | 128  | 8     | 340.18    | O3   | [mindcv_mixnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mixnet/mixnet_s_ascend.yaml) |
| mnasnet_050            | 68.07 | 512  | 8     | 367.05    | O3   | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_0.5_ascend.yaml) |
| mnasnet_075            | 71.81 | 256  | 8     | 151.02    | O0   | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_0.75_ascend.yaml) |
| mnasnet_100            | 74.28 | 256  | 8     | 153.52    | O0   | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_1.0_ascend.yaml) |
| mnasnet_130            | 75.65 | 128  | 8     | uploading | O3   | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_1.3_ascend.yaml) |
| mnasnet_140            | 76.01 | 256  | 8     | 194.9     | O0   | [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_1.4_ascend.yaml) |
| mobilenet_v1_025       | 53.87 | 64   | 8     | 75.93     | O2   | [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1/mobilenet_v1_0.25_ascend.yaml) |
| mobilenet_v1_050       | 65.94 | 64   | 8     | 51.96     | O2   | [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1/mobilenet_v1_0.5_ascend.yaml) |
| mobilenet_v1_075       | 70.44 | 64   | 8     | 57.55     | O2   | [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1/mobilenet_v1_0.75_ascend.yaml) |
| mobilenet_v1_100       | 72.95 | 64   | 8     | 44.04     | O2   | [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1/mobilenet_v1_1.0_ascend.yaml) |
| mobilenet_v2_075       | 69.98 | 256  | 8     | 169.81    | O3   | [mindcv_mobilenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2/mobilenet_v2_0.75_ascend.yaml) |
| mobilenet_v2_100       | 72.27 | 256  | 8     | 195.06    | O3   | [mindcv_mobilenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2/mobilenet_v2_1.0_ascend.yaml) |
| mobilenet_v2_140       | 75.56 | 256  | 8     | 230.06    | O3   | [mindcv_mobilenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2/mobilenet_v2_1.4_ascend.yaml) |
| mobilenet_v3_large_100 | 75.23 | 75   | 8     | 85.61     | O3   | [mindcv_mobilenetv3](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv3/mobilenet_v3_large_ascend.yaml) |
| mobilenet_v3_small_100 | 68.1  | 75   | 8     | 67.19     | O3   | [mindcv_mobilenetv3](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv3/mobilenet_v3_small_ascend.yaml) |
| mobilevit_small        | 78.47 | 64   | 8     | uploading | O3   | [mindcv_mobilevit](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilevit/mobilevit_small_ascend.yaml) |
| mobilevit_x_small      | 74.99 | 64   | 8     | uploading | O3   | [mindcv_mobilevit](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilevit/mobilevit_x_small_ascend.yaml) |
| mobilevit_xx_small     | 68.91 | 64   | 8     | uploading | O3   | [mindcv_mobilevit](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilevit/mobilevit_xx_small_ascend.yaml) |
| nasnet_a_4x1056        | 73.65 | 256  | 8     | 1562.35   | O0   | [mindcv_nasnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/nasnet/nasnet_a_4x1056_ascend.yaml) |
| pit_b                  | 81.87 | 128  | 8     | 350.33    | O2   | [mindcv_pit](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit/pit_b_ascend.yaml) |
| pit_s                  | 80.56 | 128  | 8     | 336.08    | O2   | [mindcv_pit](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit/pit_s_ascend.yaml) |
| pit_ti                 | 72.96 | 128  | 8     | 339.44    | O2   | [mindcv_pit](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit/pit_ti_ascend.yaml) |
| pit_xs                 | 78.41 | 128  | 8     | 338.7     | O2   | [mindcv_pit](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit/pit_xs_ascend.yaml) |
| poolformer_s12         | 77.33 | 128  | 8     | 316.77    | O3   | [mindcv_poolformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/poolformer/poolformer_s12_ascend.yaml) |
| pvt_large              | 81.75 | 128  | 8     | 860.41    | O2   | [mindcv_pvt](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_large_ascend.yaml) |
| pvt_medium             | 81.82 | 128  | 8     | 613.08    | O2   | [mindcv_pvt](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_medium_ascend.yaml) |
| pvt_small              | 79.66 | 128  | 8     | 431.15    | O2   | [mindcv_pvt](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_small_ascend.yaml) |
| pvt_tiny               | 74.81 | 128  | 8     | 310.74    | O2   | [mindcv_pvt](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_tiny_ascend.yaml) |
| pvt_v2_b0              | 71.5  | 128  | 8     | 338.78    | O2   | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2/pvt_v2_b0_ascend.yaml) |
| pvt_v2_b1              | 78.91 | 128  | 8     | 337.94    | O2   | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2/pvt_v2_b1_ascend.yaml) |
| pvt_v2_b2              | 81.99 | 128  | 8     | 503.79    | O2   | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2/pvt_v2_b2_ascend.yaml) |
| pvt_v2_b3              | 82.84 | 128  | 8     | 738.9     | O2   | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2/pvt_v2_b3_ascend.yaml) |
| pvt_v2_b4              | 83.14 | 128  | 8     | 1030.06   | O2   | [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2/pvt_v2_b4_ascend.yaml) |
| regnet_x_200mf         | 68.74 | 64   | 8     | 47.56     | O2   | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_200mf_ascend.yaml) |
| regnet_x_400mf         | 73.16 | 64   | 8     | 47.56     | O2   | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_400mf_ascend.yaml) |
| regnet_x_600mf         | 74.34 | 64   | 8     | 48.36     | O2   | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_600mf_ascend.yaml) |
| regnet_x_800mf         | 76.04 | 64   | 8     | 47.56     | O2   | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_800mf_ascend.yaml) |
| regnet_y_16gf          | 82.92 | 64   | 8     | uploading | O2   | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_16gf_ascend.yaml) |
| regnet_y_200mf         | 70.3  | 64   | 8     | 58.35     | O2   | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_200mf_ascend.yaml) |
| regnet_y_400mf         | 73.91 | 64   | 8     | 77.94     | O2   | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_400mf_ascend.yaml) |
| regnet_y_600mf         | 75.69 | 64   | 8     | 79.94     | O2   | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_600mf_ascend.yaml) |
| regnet_y_800mf         | 76.52 | 64   | 8     | 81.93     | O2   | [mindcv_regnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_800mf_ascend.yaml) |
| repmlp_t224            | 76.71 | 128  | 8     | 973.88    | O2   | [mindcv_repmlp](https://github.com/mindspore-lab/mindcv/blob/main/configs/repmlp/repmlp_t224_ascend.yaml) |
| repvgg_a0              | 72.19 | 32   | 8     | 27.63     | O0   | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_a0_ascend.yaml) |
| repvgg_a1              | 74.19 | 32   | 8     | 27.45     | O0   | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_a1_ascend.yaml) |
| repvgg_a2              | 76.63 | 32   | 8     | 39.79     | O0   | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_a2_ascend.yaml) |
| repvgg_b0              | 74.99 | 32   | 8     | 33.05     | O0   | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b0_ascend.yaml) |
| repvgg_b1              | 78.81 | 32   | 8     | 68.88     | O0   | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b1_ascend.yaml) |
| repvgg_b1g2            | 78.03 | 32   | 8     | 59.71     | O2   | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b1g2_ascend.yaml) |
| repvgg_b1g4            | 77.64 | 32   | 8     | 65.83     | O2   | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b1g4_ascend.yaml) |
| repvgg_b2              | 79.29 | 32   | 8     | 106.9     | O0   | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b2_ascend.yaml) |
| repvgg_b2g4            | 78.8  | 32   | 8     | 89.57     | O2   | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b2g4_ascend.yaml) |
| repvgg_b3              | 80.46 | 32   | 8     | 137.24    | O0   | [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/repvgg/repvgg_b3_ascend.yaml) |
| res2net101             | 79.56 | 32   | 8     | 49.96     | O2   | [mindcv_res2net](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_101_ascend.yaml) |
| res2net101_v1b         | 81.14 | 32   | 8     | 86.93     | O2   | [mindcv_res2net](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_101_v1b_ascend.yaml) |
| res2net50              | 79.35 | 32   | 8     | 49.16     | O2   | [mindcv_res2net](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_50_ascend.yaml) |
| res2net50_v1b          | 80.32 | 32   | 8     | 93.33     | O2   | [mindcv_res2net](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_50_v1b_ascend.yaml) |
| resnest101             | 82.9  | 128  | 8     | 719.84    | O2   | [mindcv_resnest](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnest/resnest101_ascend.yaml) |
| resnest50              | 80.81 | 128  | 8     | 376.18    | O2   | [mindcv_resnest](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnest/resnest50_ascend.yaml) |
| resnet101              | 78.24 | 32   | 8     | 50.76     | O2   | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_101_ascend.yaml) |
| resnet152              | 78.72 | 32   | 8     | 70.94     | O2   | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_152_ascend.yaml) |
| resnet18               | 70.21 | 32   | 8     | 23.98     | O2   | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_18_ascend.yaml) |
| resnet34               | 74.15 | 32   | 8     | 23.98     | O2   | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_34_ascend.yaml) |
| resnet50               | 76.69 | 32   | 8     | 31.97     | O2   | [mindcv_resnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_50_ascend.yaml) |
| resnetv2_101           | 78.48 | 32   | 8     | 56.02     | O2   | [mindcv_resnetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnetv2/resnetv2_101_ascend.yaml) |
| resnetv2_50            | 76.9  | 32   | 8     | 35.72     | O2   | [mindcv_resnetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnetv2/resnetv2_50_ascend.yaml) |
| resnext101_32x4d       | 79.83 | 32   | 8     | 68.85     | O2   | [mindcv_resnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnext/resnext101_32x4d_ascend.yaml) |
| resnext101_64x4d       | 80.3  | 32   | 8     | 112.48    | O2   | [mindcv_resnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnext/resnext101_64x4d_ascend.yaml) |
| resnext152_64x4d       | 80.52 | 32   | 8     | 157.06    | O2   | [mindcv_resnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnext/resnext152_64x4d_ascend.yaml) |
| resnext50_32x4d        | 78.53 | 32   | 8     | 50.25     | O2   | [mindcv_resnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnext/resnext50_32x4d_ascend.yaml) |
| rexnet_09              | 77.06 | 64   | 8     | 145.08    | O2   | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet/rexnet_x09_ascend.yaml) |
| rexnet_10              | 77.38 | 64   | 8     | 156.67    | O2   | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet/rexnet_x10_ascend.yaml) |
| rexnet_13              | 79.06 | 64   | 8     | 203.04    | O2   | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet/rexnet_x13_ascend.yaml) |
| rexnet_15              | 79.95 | 64   | 8     | 231.41    | O2   | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet/rexnet_x15_ascend.yaml) |
| rexnet_20              | 80.64 | 64   | 8     | 308.15    | O2   | [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet/rexnet_x20_ascend.yaml) |
| seresnet18             | 71.81 | 64   | 8     | 50.39     | O2   | [mindcv_seresnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnet18_ascend.yaml) |
| seresnet34             | 75.38 | 64   | 8     | 50.54     | O2   | [mindcv_seresnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnet34_ascend.yaml) |
| seresnet50             | 78.32 | 64   | 8     | 98.37     | O2   | [mindcv_seresnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnet50_ascend.yaml) |
| seresnext26_32x4d      | 77.17 | 64   | 8     | 73.72     | O2   | [mindcv_seresnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnext26_32x4d_ascend.yaml) |
| seresnext50_32x4d      | 78.71 | 64   | 8     | 113.82    | O2   | [mindcv_seresnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnext50_32x4d_ascend.yaml) |
| shufflenet_v1_g3_05    | 57.05 | 64   | 8     | 142.69    | O0   | [mindcv_shufflenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv1/shufflenet_v1_0.5_ascend.yaml) |
| shufflenet_v1_g3_10    | 67.77 | 64   | 8     | uploading | O0   | [mindcv_shufflenetv1](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv1/shufflenet_v1_1.0_ascend.yaml) |
| shufflenet_v2_x0_5     | 60.53 | 64   | 8     | 142.69    | O0   | [mindcv_shufflenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv2/shufflenet_v2_0.5_ascend.yaml) |
| shufflenet_v2_x1_0     | 69.47 | 64   | 8     | 267.79    | O0   | [mindcv_shufflenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv2/shufflenet_v2_1.0_ascend.yaml) |
| shufflenet_v2_x1_5     | 72.79 | 64   | 8     | 142.69    | O0   | [mindcv_shufflenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv2/shufflenet_v2_1.5_ascend.yaml) |
| shufflenet_v2_x2_0     | 75.07 | 64   | 8     | 267.79    | O0   | [mindcv_shufflenetv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv2/shufflenet_v2_2.0_ascend.yaml) |
| skresnet18             | 73.09 | 64   | 8     | 65.95     | O2   | [mindcv_skresnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/sknet/skresnet18_ascend.yaml) |
| skresnet34             | 76.71 | 32   | 8     | 43.96     | O2   | [mindcv_skresnet](https://github.com/mindspore-lab/mindcv/blob/main/configs/sknet/skresnet34_ascend.yaml) |
| skresnext50_32x4d      | 79.08 | 64   | 8     | 65.95     | O2   | [mindcv_skresnext](https://github.com/mindspore-lab/mindcv/blob/main/configs/sknet/skresnext50_32x4d_ascend.yaml) |
| squeezenet1_0          | 59.01 | 32   | 8     | 28.18     | O2   | [mindcv_squeezenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/squeezenet/squeezenet_1.0_ascend.yaml) |
| squeezenet1_1          | 58.44 | 32   | 8     | 25.58     | O2   | [mindcv_squeezenet](https://github.com/mindspore-lab/mindcv/blob/main/configs/squeezenet/squeezenet_1.1_ascend.yaml) |
| swin_tiny              | 80.82 | 256  | 8     | 1765.65   | O2   | [mindcv_swin](https://github.com/mindspore-lab/mindcv/blob/main/configs/swintransformer/swin_tiny_ascend.yaml) |
| swinv2_tiny_window8    | 81.42 | 128  | 8     | uploading | O2   | [mindcv_swinv2](https://github.com/mindspore-lab/mindcv/blob/main/configs/swintransformerv2/swinv2_tiny_window8_ascend.yaml) |
| vgg11                  | 71.86 | 32   | 8     | 61.63     | O2   | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg/vgg11_ascend.yaml) |
| vgg13                  | 72.87 | 32   | 8     | 66.47     | O2   | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg/vgg13_ascend.yaml) |
| vgg16                  | 74.61 | 32   | 8     | 73.68     | O2   | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg/vgg16_ascend.yaml) |
| vgg19                  | 75.21 | 32   | 8     | 81.13     | O2   | [mindcv_vgg](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg/vgg19_ascend.yaml) |
| visformer_small        | 81.76 | 64   | 8     | 155.88    | O3   | [mindcv_visformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer/visformer_small_ascend.yaml) |
| visformer_small_v2     | 82.17 | 64   | 8     | 158.27    | O3   | [mindcv_visformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer/visformer_small_v2_ascend.yaml) |
| visformer_tiny         | 78.28 | 128  | 8     | 393.29    | O3   | [mindcv_visformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer/visformer_tiny_ascend.yaml) |
| visformer_tiny_v2      | 78.82 | 256  | 8     | 627.2     | O3   | [mindcv_visformer](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer/visformer_tiny_v2_ascend.yaml) |
| vit_b_32_224           | 75.86 | 256  | 8     | 623.09    | O2   | [mindcv_vit](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit/vit_b32_224_ascend.yaml) |
| vit_l_16_224           | 76.34 | 48   | 8     | 613.98    | O2   | [mindcv_vit](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit/vit_l16_224_ascend.yaml) |
| vit_l_32_224           | 73.71 | 128  | 8     | 527.58    | O2   | [mindcv_vit](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit/vit_l32_224_ascend.yaml) |
| volo_d1                | 82.59 | 128  | 8     | 575.54    | O3   | [mindcv_volo](https://github.com/mindspore-lab/mindcv/blob/main/configs/volo/volo_d1_ascend.yaml) |
| xception               | 79.01 | 32   | 8     | 98.03     | O2   | [mindcv_xception](https://github.com/mindspore-lab/mindcv/blob/main/configs/xception/xception_ascend.yaml) |
| xcit_tiny_12_p16_224   | 77.67 | 128  | 8     | 352.3     | O2   | [mindcv_xcit](https://github.com/mindspore-lab/mindcv/blob/main/configs/xcit/xcit_tiny_12_p16_ascend.yaml) |

### Object Detection

#### Accuracies are reported on COCO2017

| model | map | bs | cards | ms/step | amp | config
:-: | :-: | :-: | :-: | :-: | :-: | :-: |
| yolov8_n | 37.2 | 16 | 8 |  302  |  O0 |   [mindyolo_yolov8](https://github.com/mindspore-lab/mindyolo/tree/main/configs/yolov8) |
| yolov8_s | 44.6 | 16 | 8 |  uploading  |  O0 |   [mindyolo_yolov8](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov8_m | 50.5 | 16 | 8 |  454  |  O0 |   [mindyolo_yolov8](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov8_l | 52.8 | 16 | 8 |  536  |  O0 |   [mindyolo_yolov8](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov8_x | 53.7 | 16 | 8 |  636  |  O0 |   [mindyolo_yolov8](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov7_t| 37.5 | 16 | 8 |  594.91  |  O0 |   [mindyolo_yolov7](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) | 
| yolov7_l| 50.8 | 16 | 8 |  905.26  |  O0 |   [mindyolo_yolov7](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |
| yolov7_x|  52.4| 16 | 8 |  819.36  |  O0 |   [mindyolo_yolov7](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |
| yolov5_n | 27.3 | 32 | 8 |  504.68  |  O3 |   [mindyolo_yolov5](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov5_s | 37.6 | 32 | 8 |  535.32  |  O3 |   [mindyolo_yolov5](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov5_m | 44.9 | 32 | 8 |  646.75  |  O3 |   [mindyolo_yolov5](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov5_l | 48.5 | 32 | 8 |  684.81  |  O3 |   [mindyolo_yolov5](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov5_x | 50.5 | 16 | 8 |  613.81  |  O0 |   [mindyolo_yolov5](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov4_csp | 45.4 | 16 | 8 |  709.70  |  O0 |   [mindyolo_yolov4](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov4) |
| yolov4_csp(silu) | 45.8 | 16 | 8 |  594.97  |  O0 |   [mindyolo_yolov4](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov4) |
| yolov3_darknet53| 45.5 | 16 | 8 |  481.37  |  O0 |   [mindyolo_yolov3](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov3) |
| yolox_n | 24.1 | 8 | 8 |  201  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_t | 33.3 | 8 | 8 |  190  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_s | 40.7 | 8 | 8 |  270  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_m | 46.7 | 8 | 8 |  311  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_l | 49.2 | 8 | 8 |  535  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_x | 51.6 | 8 | 8 |  619  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_darknet53 | 47.7 | 8 | 8 |  411  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| fasterrcnn_r50_fpn | 37.0 | 2 | 8 |  130  |  O2 |   [fasterrcnn_r50_fpn](https://github.com/mindspore-lab/models/tree/master/official/cv/RCNN) |

### Semantic Segmentation

#### Accuracies are reported on CityScapes

| model | miou | bs | cards | ms/step | amp | config
:-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ocrnet_hw48 | 82.96 | 2 | 8 |  220.48  |  O3 |   [ocrnet_hw48](https://github.com/mindspore-lab/models/tree/master/official/cv/OCRNet/config/ocrnet/config_ocrnet_hrw48_16k.yml) |
| ocrnet_hw32 | 82.27 | 2 | 8 |  168.6  |  O3 |   [ocrnet_hw32](https://github.com/mindspore-lab/models/tree/master/official/cv/OCRNet/config/ocrnet/config_ocrnet_hrw32_16k.yml) |

### OCR

##### Training Performance on Ascend 910A

#### Text Detection

| model  |dataset |bs | cards | fscore | ms/step | fps | amp | config |
:-:     |   :-:   | :-: | :-: |  :-:   |  :-:    | :-:  |  :-: |  :-:    |
| dbnet_mobilenetv3  | icdar2015  | 10 | 1 | 77.23 | 100 | 100 | O0 | [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet_resnet18     | icdar2015  | 20 | 1 | 81.73 | 186 | 108 | O0 | [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet_resnet50     | icdar2015  | 10 | 1 | 85.05 | 133 | 75.2 | O0 | [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet++_resnet50   | icdar2015  | 32 | 1 | 86.74 | 571 | 56 | O0 | [mindocr_dbnet++](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| psenet_resnet152   | icdar2015  | 8 | 8 | 82.06 | 8455.88 | 7.57 | O0 | [mindocr_psenet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) |
| east_resnet50      | icdar2015  | 20 | 8 | 84.87 | 256 | 625 | O0 | [mindocr_east](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east)   |
| fcenet_resnet50    | icdar2015  | 8 | 4 | 84.12 | 4570.64 | 7 | O0 | [mindocr_fcenet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/fcenet)   |

#### Text Recognition

| model  |dataset |bs | cards | acc | ms/step | fps | amp | config |
:-:     |   :-:   | :-: | :-: |  :-:   |  :-:    | :-:  |  :-: |  :-:    |
| svtr_tiny          | IC03,13,15,IIIT,etc | 512 | 4 | 89.02 | 690 | 2968 | O2 |  [mindocr_svtr](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/svtr)   |
| crnn_vgg7          | IC03,13,15,IIIT,etc | 16 | 8 | 82.03 | 22.06 | 5802.71 | O3 |  [mindocr_crnn](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |
| crnn_resnet34_vd   | IC03,13,15,IIIT,etc | 64 | 8 | 84.45 | 76.48 | 6694.84 | O3 |  [mindocr_crnn](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |
| rare_resnet34_vd   | IC03,13,15,IIIT,etc | 512 | 4 | 85.19 | 449 | 4561 | O2 |  [mindocr_rare](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare)   |

#### Text Direction Classification

| model  |dataset |bs | cards | acc | ms/step | fps | amp | config |
:-:     |   :-:   | :-: | :-: |  :-:   |  :-:    | :-:  |  :-: |  :-:    |
| mobilenetv3 | RCTW17,MTWI,LSVT | 256 | 4 | 94.59 | 172.9 | 5923.5 | O0 | [mindocr_mobilenetv3](https://github.com/mindspore-lab/mindocr/tree/main/configs/cls/mobilenetv3)   |


##### Inference Performance on Ascend 310P for mindspore models

#### Text Detection

|       model       |  dataset  | fscore |  fps  |                        mindocr recipe                        |
| :---------------: | :-------: | :-----: | :---: | :----------------------------------------------------------: |
| dbnet_mobilenetv3 | icdar2015 | 0.7696  | 26.19 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet) |
|  dbnet_resnet18   | icdar2015 | 0.8173  | 24.04 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet) |
|  dbnet_resnet50   | icdar2015 | 0.8500  | 21.69 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet) |
| dbnet++_resnet50  | icdar2015 | 0.8679  | 8.46  | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet) |
| psenet_resnet152  | icdar2015 | 0.8250  | 2.31  | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) |
|   east_resnet50   | icdar2015 | 0.8686  | 6.72  | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east) |

#### Text Recognition

|      model       |  dataset  |  acc   |  fps   |                        mindocr recipe                        |
| :--------------: | :-------: | :----: | :----: | :----------------------------------------------------------: |
|    crnn_vgg7     | icdar2015 | 0.6601 | 465.64 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn) |
| crnn_resnet34_vd | icdar2015 | 0.6967 | 397.29 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn) |
| rare_resnet34_vd | icdar2015 | 0.6947 | 273.23 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare) |


##### Inference Performance on Ascend 310P for paddleocr/mmocr models via onnx

#### Text Detection

|             name              |       dataset  | fscore |  fps  |                            mindocr recipe                           |
| :---------------------------: | :-------: | :-----: | :---: | :----------------------------------------------------------: |
|   ch_ppocr_server_v2.0_det    |   MLT17   | 0.4622  | 21.65 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|        ch_PP-OCRv3_det        |   MLT17   | 0.3389  | 22.40 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md)|
|        ch_PP-OCRv2_det        |  MLT17   | 0.4299  | 21.90 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
| ch_ppocr_mobile_slim_v2.0_det | MLT17   | 0.3166  | 19.88 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md)|
|   ch_ppocr_mobile_v2.0_det    |   MLT17   | 0.3156  | 21.96 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|        en_PP-OCRv3_det        |  icdar2015 | 0.4214  | 55.55 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|        ml_PP-OCRv3_det        |    MLT17   | 0.6601  | 22.48 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|       dbnet_resnet50_vd       | icdar2015 | 0.7989  | 21.17 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|      psenet_resnet50_vd       | icdar2015 | 0.8044  | 7.75  | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md)|
|       east_resnet50_vd        |   icdar2015 | 0.8558  | 20.70 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|       sast_resnet50_vd        |   icdar2015 | 0.8177  | 22.14 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|       dbnet++_resnet50        |  icdar2015 | 0.8136  | 10.66 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|        fcenet_resnet50        |  icdar2015 | 0.8367  | 3.34  | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md)|

#### Text Recognition

|              name              |        dataset        |  acc   |  fps  |                           mindocr recipe                     |
| :----------------------------: |  :------------------: | :----: | :----: | :----------------------------------------------------------: |
|    ch_ppocr_server_v2.0_rec    |   MLT17 (ch) | 0.4991 | 154.16 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|        ch_PP-OCRv3_rec         |   MLT17 (ch) | 0.4991 | 408.38 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|        ch_PP-OCRv2_rec         |   MLT17 (ch) | 0.4459 | 203.34 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|    ch_ppocr_mobile_v2.0_rec    |   MLT17 (ch) | 0.2459 | 167.67 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|        en_PP-OCRv3_rec         |   MLT17 (en) | 0.7964 | 917.01 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
| en_number_mobile_slim_v2.0_rec |   MLT17 (en) | 0.0164 | 445.04 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|   en_number_mobile_v2.0_rec    |   MLT17 (en) | 0.4304 | 458.66 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|        crnn_resnet34_vd        |  icdar2015       | 0.6635 | 420.80 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|      rosetta_resnet34_vd       |  icdar2015       | 0.6428 | 552.40 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md)|
|             vitstr             |  icdar2015       | 0.6842 | 364.67 | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
|         nrtr_resnet31          |  icdar2015       | 0.6726 | 32.63  | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md)|
|        satrn_shallowcnn        |  icdar2015       | 0.7352 | 32.14  | [config](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/models_list_thirdparty.md) |
