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
'''DSAN model'''
import mindspore.nn as nn
import models.ResNet as ResNet
import lmmd

class DSAN(nn.Cell):

    def __init__(self, num_classes=31, bottle_neck=True):
        super(DSAN, self).__init__()
        self.feature_layers = ResNet.resnet50(pretrained=True)
        self.lmmd_loss = lmmd.LMMD_loss(class_num=num_classes)
        self.bottle_neck = bottle_neck
        if bottle_neck:
            self.bottle = nn.Dense(2048, 256)
            self.cls_fc = nn.Dense(256, num_classes)
        else:
            self.cls_fc = nn.Dense(2048, num_classes)


    def construct(self, source, target):
        source = self.feature_layers(source)
        if self.bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)
        target = self.feature_layers(target)
        if self.bottle_neck:
            target = self.bottle(target)
        t_pre = self.cls_fc(target)
        return s_pred, t_pre, source, target

    def predict(self, x):
        x = self.feature_layers(x)
        if self.bottle_neck:
            x = self.bottle(x)
        return self.cls_fc(x)
    
class DSAN_for_export(nn.Cell):

    def __init__(self, num_classes=31, bottle_neck=True):
        super(DSAN_for_export, self).__init__()
        self.feature_layers = ResNet.resnet50()
        self.lmmd_loss = lmmd.LMMD_loss(class_num=num_classes)
        self.bottle_neck = bottle_neck
        if bottle_neck:
            self.bottle = nn.Dense(2048, 256)
            self.cls_fc = nn.Dense(256, num_classes)
        else:
            self.cls_fc = nn.Dense(2048, num_classes)


    def construct(self, x):
        x = self.feature_layers(x)         
        if self.bottle_neck:            
             x = self.bottle(x)         
        return self.cls_fc(x)
