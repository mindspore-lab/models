# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model checkpoint urls."""

model_urls = {
    # lenet series
    "lenet": "https://download.mindspore.cn/vision/classification/lenet_mnist.ckpt",
    # resnet series
    "resnet18": "https://download.mindspore.cn/vision/classification/resnet18_224.ckpt",
    "resnet34": "https://download.mindspore.cn/vision/classification/resnet34_224.ckpt",
    "resnet50": "https://download.mindspore.cn/vision/classification/resnet50_224.ckpt",
    "resnet101": "https://download.mindspore.cn/vision/classification/resnet101_224.ckpt",
    "resnet152": "https://download.mindspore.cn/vision/classification/resnet152_224.ckpt",
    # mobilenet_v2 series
    "mobilenet_v2_1.4_224": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.4_224.ckpt",
    "mobilenet_v2_1.3_224": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.3_224.ckpt",
    "mobilenet_v2_1.0_224": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_224.ckpt",
    "mobilenet_v2_1.0_192": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_192.ckpt",
    "mobilenet_v2_1.0_160": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_160.ckpt",
    "mobilenet_v2_1.0_128": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_128.ckpt",
    "mobilenet_v2_1.0_96": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_96.ckpt",
    "mobilenet_v2_0.75_224": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.75_224.ckpt",
    "mobilenet_v2_0.75_192": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.75_192.ckpt",
    "mobilenet_v2_0.75_160": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.75_160.ckpt",
    "mobilenet_v2_0.75_128": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.75_128.ckpt",
    "mobilenet_v2_0.75_96": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.75_96.ckpt",
    "mobilenet_v2_0.5_224": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.5_224.ckpt",
    "mobilenet_v2_0.5_192": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.5_192.ckpt",
    "mobilenet_v2_0.5_160": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.5_160.ckpt",
    "mobilenet_v2_0.5_128": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.5_128.ckpt",
    "mobilenet_v2_0.5_96": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.5_96.ckpt",
    "mobilenet_v2_0.35_224": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.35_224.ckpt",
    "mobilenet_v2_0.35_192": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.35_192.ckpt",
    "mobilenet_v2_0.35_160": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.35_160.ckpt",
    "mobilenet_v2_0.35_128": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.35_128.ckpt",
    "mobilenet_v2_0.35_96": "https://download.mindspore.cn/vision/classification/mobilenet_v2_0.35_96.ckpt",
    # ViT series
    "vit_b_16_224": "https://download.mindspore.cn/vision/classification/vit_b_16_224.ckpt",
    "vit_b_16_384": "https://download.mindspore.cn/vision/classification/vit_b_16_384.ckpt",
    "vit_l_16_224": "https://download.mindspore.cn/vision/classification/vit_l_16_224.ckpt",
    "vit_l_16_384": "https://download.mindspore.cn/vision/classification/vit_l_16_384.ckpt",
    "vit_b_32_224": "https://download.mindspore.cn/vision/classification/vit_b_32_224_tv.ckpt",
    "vit_b_32_384": "https://download.mindspore.cn/vision/classification/vit_b_32_384.ckpt",
    "vit_l_32_224": "https://download.mindspore.cn/vision/classification/vit_l_32_224_tv.ckpt",
    # EfficientNet series
    # eg. b0(Model name)_1(Width Coefficient)_1(Depth Coefficient)_224(Resolution)_0.2(Dropout Rate).ckpt
    "efficientnet_b0": "https://download.mindspore.cn/vision/classification/efficientnet_b0_1_1_224_0.2.ckpt",
    "efficientnet_b1": "https://download.mindspore.cn/vision/classification/efficientnet_b1_1_1.1_240_0.2.ckpt",
    "efficientnet_b2": "https://download.mindspore.cn/vision/classification/efficientnet_b2_1.1_1.2_260_0.3.ckpt",
    "efficientnet_b3": "https://download.mindspore.cn/vision/classification/efficientnet_b3_1.2_1.4_300_0.3.ckpt",
    "efficientnet_b4": "https://download.mindspore.cn/vision/classification/efficientnet_b4_1.4_1.8_380_0.4.ckpt",
    "efficientnet_b5": "https://download.mindspore.cn/vision/classification/efficientnet_b5_1.6_2.2_456_0.4.ckpt",
    "efficientnet_b6": "https://download.mindspore.cn/vision/classification/efficientnet_b6_1.8_2.6_528_0.5.ckpt",
    "efficientnet_b7": "https://download.mindspore.cn/vision/classification/efficientnet_b7_2.0_3.1_600_0.5.ckpt",
}
