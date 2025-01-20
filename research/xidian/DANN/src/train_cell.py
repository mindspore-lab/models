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

import mindspore.nn as nn
from mindspore.ops import stop_gradient
import mindspore.ops as ops
from mindspore import ParameterTuple
import mindspore as ms

class withlosscell_class(nn.Cell):
    def __init__(self, backbone, classifier, class_loss):
        super(withlosscell_class, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.class_loss = class_loss

    def construct(self, img, cls_label, weightclass):
        feature = self.backbone(img)
        cls_output = self.classifier(feature)
        err_cls, _ = self.class_loss(cls_output, cls_label, weightclass)
        return err_cls


class TrainOneStep_cls(nn.Cell):
    def __init__(self, network, optimizer_backbone, optimizer_cls):
        super(TrainOneStep_cls, self).__init__(auto_prefix=False)
        self.network = network
        self.optimizer_backbone = optimizer_backbone
        self.optimizer_cls = optimizer_cls
        self.weights_backbone = ParameterTuple(self.optimizer_backbone.parameters)
        self.weights_cls = ParameterTuple(self.optimizer_cls.parameters)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        out = self.network(*inputs)
        grads_b = self.grad(self.network, self.weights_backbone)(*inputs)
        grads_c = self.grad(self.network, self.weights_cls)(*inputs)
        out = ops.functional.depend(out, self.optimizer_backbone(grads_b))
        out = ops.functional.depend(out, self.optimizer_cls(grads_c))
        return out


class withlosscell_domain(nn.Cell):
    def __init__(self, backbone_s, backbone_t,domain_classifier, domain_loss):
        super(withlosscell_domain, self).__init__()
        self.backbone_s = backbone_s
        self.backbone_t = backbone_t
        self.domain_classifier = domain_classifier
        self.domain_loss = domain_loss

    def construct(self, img_s, dlabel_s, img_t, dlabel_t, weightdomain):
        feature_s = self.backbone_s(img_s)
        feature_t = self.backbone_t(img_t)
        feature_s, feature_t = stop_gradient(feature_s), stop_gradient(feature_t)
        domain_output_s = self.domain_classifier(feature_s)
        domain_output_t = self.domain_classifier(feature_t)
        err_s, _ = self.domain_loss(domain_output_s, dlabel_s, weightdomain)
        err_t, _ = self.domain_loss(domain_output_t, dlabel_t, weightdomain)
        err = err_s + err_t
        return err


class TrainOneStepDomain(nn.Cell):
    def __init__(self, network, optimizer_domain):
        super(TrainOneStepDomain, self).__init__(auto_prefix=False)
        self.network = network
        self.optimizer_domain = optimizer_domain
        self.weights_domain = ParameterTuple(self.optimizer_domain.parameters)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        out = self.network(*inputs)
        grads_d = self.grad(self.network, self.weights_domain)(*inputs)
        out = ops.functional.depend(out, self.optimizer_domain(grads_d))
        return out


class withlosscell_d(nn.Cell):
    def __init__(self, backbone, domain_classifier, domain_loss):
        super(withlosscell_d, self).__init__()
        self.backbone = backbone
        self.domain_classifier = domain_classifier
        self.domain_loss = domain_loss

    def construct(self, img_t, dlabel_t, weightdomain):
        feature_t = self.backbone(img_t)
        domain_output_t = self.domain_classifier(feature_t)
        err_t, _ = self.domain_loss(domain_output_t, dlabel_t, weightdomain)
        return err_t


class TrainOneStepD(nn.Cell):
    def __init__(self, network, optimizer_backbone, optimizer_domain):
        super(TrainOneStepD, self).__init__(auto_prefix=False)
        self.network = network
        self.optimizer_backbone = optimizer_backbone
        self.optimizer_domain = optimizer_domain
        self.weights_backbone = ParameterTuple(self.optimizer_backbone.parameters)
        self.weights_domain = ParameterTuple(self.optimizer_domain.parameters)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        out = self.network(*inputs)
        grads_b = self.grad(self.network, self.weights_backbone)(*inputs)
        grads_d = self.grad(self.network, self.weights_domain)(*inputs)
        out = ops.functional.depend(out, self.optimizer_backbone(grads_b))
        out = ops.functional.depend(out, self.optimizer_domain(grads_d))
        return out

class withEvalCell(nn.Cell):
    def __init__(self, backbone, classifier, loss):
        super(withEvalCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.classifier = classifier
        self.loss = loss

    def construct(self, x, lable, weightClass):
        features = self.backbone(x)
        class_output = self.classifier(features)
        pre = ops.Argmax(output_type=ms.int32)(class_output)
        err, _ = self.loss(class_output, lable, weightClass)
        return err, (pre)
