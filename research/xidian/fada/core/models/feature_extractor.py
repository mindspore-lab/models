# Copyright 2023 Xidian University
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

import mindspore
from mindspore import nn
from . import vgg
from .resnet import Resnet_101


class vgg_feature_extractor(nn.Cell):
    def __init__(self, backbone_name, pretrained_weights=None, aux=False, pretrained_backbone=True, freeze_bn=False):
        super(vgg_feature_extractor, self).__init__()
        backbone = vgg.__dict__[backbone_name](
                pretrained=pretrained_backbone, pretrained_weights=pretrained_weights)
            
        features, _ = list(backbone.features.children()), list(backbone.classifier.children())

        #remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))
        for i in [23,25,27]:
            features[i].dilation = (2, 2)
            features[i].padding = (2, 2)
        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        backbone = nn.Sequential(*([features[i] for i in range(len(features))] + [fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
        return_layers = {'4': 'low_fea', '32': 'out'}
        
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x):
        feas = self.backbone(x)
        out = feas['out']
        return out


class resnet_feature_extractor(nn.Cell):
    def __init__(self, cfg, backbone_name, pretrained=True, freeze_bn=False):

        super(resnet_feature_extractor, self).__init__()

        self.backbone = Resnet_101(cfg, pretrained=pretrained, freeze_bn=freeze_bn)

    def forward(self, x):
        out = self.backbone(x)['out']
        return out