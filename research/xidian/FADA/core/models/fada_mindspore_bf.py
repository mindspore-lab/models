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
import mindspore.nn as nn
from mindspore import ops

from core.models import build_feature_extractor, build_classifier
from core.models.resnet import Resnet_101
from core.models.classifier import ASPP_Classifier_V2
from core.models.discriminator import PixelDiscriminator
import mindspore.ops.functional as F


def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape

    loss = -1 * ops.Cast()(soft_label, mindspore.float32) * nn.LogSoftmax(axis=1)(pred)

    if pixel_weights is None:
        return ops.ReduceMean()(ops.ReduceSum()(loss, 1))
    return ops.ReduceMean()(pixel_weights * ops.ReduceSum()(loss, 1))


class FADA_MindSpore_Feature_Extractor(nn.Cell):

    def __init__(self, cfg):

        super(FADA_MindSpore_Feature_Extractor, self).__init__()
        model_name, backbone_name = cfg.MODEL.NAME.split('_')
        if backbone_name.startswith('resnet'):
            self.feature_extractor = Resnet_101(cfg, pretrained=True, freeze_bn=cfg.MODEL.FREEZE_BN)
        elif backbone_name.startswith('vgg'):
            # self.feature_extractor = vgg_feature_extractor(backbone_name, pretrained_weights=cfg.MODEL.WEIGHTS, aux=False,
            #                                  pretrained_backbone=True, freeze_bn=cfg.MODEL.FREEZE_BN)
            pass
        else:
            raise NotImplementedError

    def construct(self, inputs):

        feature = self.feature_extractor(inputs)

        return feature


class FADA_MindSpore_Classifier(nn.Cell):

    def __init__(self, cfg):

        super(FADA_MindSpore_Classifier, self).__init__()

        _, backbone_name = cfg.MODEL.NAME.split('_')

        if backbone_name.startswith('vgg'):
            self.classifier = ASPP_Classifier_V2(1024, [6, 12, 18, 24], [6, 12, 18, 24], cfg.MODEL.NUM_CLASSES)
        elif backbone_name.startswith('resnet'):
            self.classifier = ASPP_Classifier_V2(2048, [6, 12, 18, 24], [6, 12, 18, 24], cfg.MODEL.NUM_CLASSES)
        else:
            raise NotImplementedError

    def construct(self, feature, size):

        outputs = self.classifier(feature, size)

        return outputs


class FADA_MindSpore_Discriminator(nn.Cell):

    def __init__(self, cfg, num_features=None, mid_nc=256):

        super(FADA_MindSpore_Discriminator, self).__init__()

        _, backbone_name = cfg.MODEL.NAME.split('_')
        if backbone_name.startswith('vgg'):
            if num_features is None:
                num_features = 1024
            self.discriminator = PixelDiscriminator(num_features, mid_nc, num_classes=cfg.MODEL.NUM_CLASSES)
        elif backbone_name.startswith('resnet'):
            if num_features is None:
                num_features = 2048
            self.discriminator = PixelDiscriminator(num_features, mid_nc, num_classes=cfg.MODEL.NUM_CLASSES)
        else:
            raise NotImplementedError

    def construct(self, feature, size):

        outputs = self.discriminator(feature, size)

        return outputs


class WithLossCellSrc(nn.Cell):

    def __init__(self, feature_extractor, classifier, loss_fn):
        super(WithLossCellSrc, self).__init__(auto_prefix=True)
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.loss_fn = loss_fn

    def construct(self, data, label):
        size = label.shape[-2:]

        feature = self.feature_extractor(data)

        out = self.classifier(feature, size)

        loss = self.loss_fn(out, label)

        return loss


class WithLossCellG(nn.Cell):
    """连接生成器和损失"""

    def __init__(self, feature_extractor, classifier, discriminator, loss_fn):
        super(WithLossCellG, self).__init__(auto_prefix=True)
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.discriminator = discriminator
        self.loss_fn = loss_fn
        self.sftmx = ops.Softmax(axis=1)
        self.div = ops.Div()

    def construct(self, src_input, tgt_input, src_label, src_size, tgt_size):

        src_fea = self.feature_extractor(src_input)

        src_pred = self.classifier(src_fea, src_size)
        temperature = 1.8
        src_pred = self.div(src_pred, temperature)
        loss_seg = self.loss_fn(src_pred, src_label)
        
        # no gard
        
        src_soft_label = self.sftmx(src_pred)
        src_soft_label = ops.stop_gradient(src_soft_label)
        src_soft_label[src_soft_label > 0.9] = 0.9
        
        # end
        
        tgt_fea = self.feature_extractor(tgt_input)

        # no gard

        tgt_pred = self.classifier(tgt_fea, tgt_size)
        tgt_pred = self.div(tgt_pred, temperature)
        tgt_soft_label = self.sftmx(tgt_pred)
        tgt_soft_label = ops.stop_gradient(tgt_soft_label)
        tgt_soft_label[tgt_soft_label > 0.9] = 0.9

        # end

        tgt_D_pred = self.discriminator(tgt_fea, tgt_size)
        loss_adv_tgt = 0.001 * soft_label_cross_entropy(tgt_D_pred,
                                                        ops.Concat(axis=1)(
                                                            (tgt_soft_label, ops.ZerosLike()(tgt_soft_label))))

        src_fea = ops.stop_gradient(src_fea)
        tgt_fea = ops.stop_gradient(tgt_fea)
        tgt_soft_label = ops.stop_gradient(tgt_soft_label)
        src_soft_label = ops.stop_gradient(src_soft_label)
        return loss_seg, loss_adv_tgt, src_fea, tgt_fea, src_soft_label, tgt_soft_label


class WithLossCellD(nn.Cell):
    """连接判别器和损失"""

    def __init__(self, feature_extractor, classifier, discriminator):
        super(WithLossCellD, self).__init__(auto_prefix=True)
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.discriminator = discriminator
        self.sftmx = ops.Softmax(axis=1)
        self.div = ops.Div()
        
    def construct(self, src_fea, tgt_fea, src_soft_label, tgt_soft_label, src_size, tgt_size):
        """构建判别器损失计算结构"""

        src_fea = ops.stop_gradient(src_fea)
        tgt_fea = ops.stop_gradient(tgt_fea)
        tgt_soft_label = ops.stop_gradient(tgt_soft_label)
        src_soft_label = ops.stop_gradient(src_soft_label)
        
        src_D_pred = self.discriminator(src_fea, src_size)

        loss_D_src = 0.5 * soft_label_cross_entropy(src_D_pred,
                                                    ops.Concat(axis=1)(
                                                        (src_soft_label, ops.ZerosLike()(src_soft_label))))

        tgt_D_pred = self.discriminator(tgt_fea, tgt_size)
        loss_D_tgt = 0.5 * soft_label_cross_entropy(tgt_D_pred,
                                                    ops.Concat(axis=1)(
                                                        (ops.ZerosLike()(tgt_soft_label), tgt_soft_label)))

        return loss_D_src, loss_D_tgt


class WithEvalCellSrc(nn.Cell):

    def __init__(self, feature_extractor, classifier):
        super(WithEvalCellSrc, self).__init__(auto_prefix=True)
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def construct(self, data, label):
        size = label.shape[-2:]

        feature = self.feature_extractor(data)

        outputs = self.classifier(feature, size)

        return outputs


class CustomTrainOneStepCellSrc(nn.Cell):

    def __init__(self, netG, optimizer_f, optimizer_c):
        super(CustomTrainOneStepCellSrc, self).__init__(auto_prefix=True)

        self.netG = netG
        self.optimizer_f = optimizer_f
        self.optimizer_c = optimizer_c
        self.weights_f = self.optimizer_f.parameters
        self.weights_c = self.optimizer_c.parameters
        self.grad = ops.GradOperation(get_by_list=True)
        self.cast_long = ops.Cast()

    def construct(self, *inputs):
        loss = self.netG(*inputs)
        grads_f = self.grad(self.netG, self.weights_f)(*inputs)
        grads_c = self.grad(self.netG, self.weights_c)(*inputs)
        loss = F.depend(loss, self.optimizer_f(grads_f))

        return F.depend(loss, self.optimizer_c(grads_c))


class CustomTrainOneStepCellG(nn.Cell):

    def __init__(self, netG, optimizer_f, optimizer_c):
        super(CustomTrainOneStepCellG, self).__init__(auto_prefix=True)

        self.netG = netG
        self.optimizer_f = optimizer_f
        self.optimizer_c = optimizer_c
        self.weights_f = self.optimizer_f.parameters
        self.weights_c = self.optimizer_c.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss_seg, loss_adv_tgt, src_fea, tgt_fea, src_soft_label, tgt_soft_label = self.netG(*inputs)
        grads_f = self.grad(self.netG, self.weights_f)(*inputs)
        grads_c = self.grad(self.netG, self.weights_c)(*inputs)

        loss_adv_tgt = F.depend(loss_adv_tgt, self.optimizer_f(grads_f))

        loss_adv_tgt = F.depend(loss_adv_tgt, self.optimizer_c(grads_c))
        return loss_seg, loss_adv_tgt, src_fea, tgt_fea, src_soft_label, tgt_soft_label 


class CustomTrainOneStepCellD(nn.Cell):

    def __init__(self, netD, optimizer_d):
        super(CustomTrainOneStepCellD, self).__init__(auto_prefix=True)

        self.netD = netD
        self.optimizer_d = optimizer_d
        self.weights_d = self.optimizer_d.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):

        loss_D_src, loss_D_tgt = self.netD(*inputs)
        grads_d = self.grad(self.netD, self.weights_d)(*inputs)
        loss_D_tgt = F.depend(loss_D_tgt, self.optimizer_d(grads_d))

        return loss_D_src, loss_D_tgt
