# Copyright 2023 Huawei Technologies Co., Ltd
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
"""functions of criterion"""
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import (
    functional,
    operations,
    LogSoftmax,
    KLDivLoss,
    Size
)

from src.model.factory import create_teacher_model

class SoftTargetCrossEntropy(LossBase):
    """SoftTargetCrossEntropy for MixUp Augment"""

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()
        self.mean_ops = operations.ReduceMean(keep_dims=False)
        self.sum_ops = operations.ReduceSum(keep_dims=False)
        self.log_softmax = operations.LogSoftmax()

    def construct(self, logits, labels):
        logits = operations.Cast()(logits, mstype.float32)
        labels = operations.Cast()(labels, mstype.float32)
        loss = self.sum_ops((-1 * labels) * self.log_softmax(logits), -1)
        return self.mean_ops(loss)


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""

    def __init__(self, sparse=True, reduction='mean',
                 smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = operations.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(
            1.0 * smooth_factor / (num_classes - 1), mstype.float32
        )
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)
        self.cast = ops.Cast()

    def construct(self, logits, labels):
        if self.sparse:
            labels = self.onehot(
                labels, functional.shape(logits)[1],
                self.on_value, self.off_value
            )
        labels = operations.Cast()(labels, mstype.float32)
        logits = operations.Cast()(logits, mstype.float32)
        loss2 = self.ce(logits, labels)
        return loss2


class DistillationLoss(LossBase):
    """
    This module wraps a standard criterion and adds an extra knowledge
    distillation loss by taking a teacher model prediction and
    using it as additional supervision.
    """
    def __init__(self, base_criterion: LossBase, teacher_model: nn.Cell,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

        self.kl_div = KLDivLoss()
        self.log_softmax = LogSoftmax(axis=1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def construct(self, inputs, outputs, labels):
        """
        Args:
        inputs: The original inputs that are feed to the teacher model
        outputs: the outputs of the model to be trained. It is expected to be
            either a Tensor, or a Tuple[Tensor, Tensor], with the original output
            in the first position and the distillation predictions as the second output
        labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, ms.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)

        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")

        teacher_outputs = self.teacher_model(inputs)
        dist_loss = 0.0
        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            dist_loss = self.kl_div(
                self.log_softmax(outputs_kd / T),
                #We provide the teacher's targets in log probability because we use log_target=True
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                self.log_softmax(teacher_outputs / T),
                reduction='sum',
            ) * (T * T) / Size()(outputs_kd)
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            #But we also experiments output_kd.size(0)
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            dist_loss = self.cross_entropy(
                outputs_kd, teacher_outputs.argmax(axis=1)
            )

        loss = base_loss * (1 - self.alpha) + dist_loss * self.alpha
        return loss


def get_criterion_by_args(args):
    criterion = get_criterion(
        smoothing=args.smoothing,
        num_classes=args.num_classes,
        mixup=args.mixup,
        cutmix=args.cutmix,
        cutmix_minmax=args.cutmix_minmax,
        bce_loss=args.bce_loss,
        distillation_type=args.distillation_type,
        teacher_path=args.teacher_path,
        teacher_model=args.teacher_model,
        distillation_alpha=args.distillation_alpha,
        distillation_tau=args.distillation_tau
    )
    return criterion

def get_criterion(
        smoothing,
        num_classes,
        mixup,
        cutmix,
        cutmix_minmax,
        bce_loss,
        distillation_type,
        teacher_path,
        teacher_model,
        distillation_alpha,
        distillation_tau
):
    """Get criterion function"""
    assert smoothing >= 0
    assert smoothing <= 1.

    mixup_active = False
    if mixup > 0:
        mixup_active = True
    if cutmix > 0:
        mixup_active = True
    if cutmix_minmax is not None:
        mixup_active = True

    if mixup_active:
        # smoothing is handled with mixup label transform
        print(25 * "=" + "Using MixBatch" + 25 * "=")
        criterion = SoftTargetCrossEntropy()
    elif smoothing:
        print(25 * "=" + "Using label smoothing" + 25 * "=")
        criterion = CrossEntropySmooth(sparse=True, reduction="mean",
                                       smooth_factor=smoothing,
                                       num_classes=num_classes)
    else:
        criterion = nn.SoftmaxCrossEntropyWithLogits()

    if bce_loss:
        criterion = nn.BCEWithLogitsLoss()

    teacher_net = None
    if distillation_type != 'none':
        assert teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {teacher_model}")
        teacher_net = create_teacher_model(
            teacher_model,
            checkpoint_path=teacher_path,
        )
        teacher_net.set_train(False)

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if distillation_type is 'none'
    criterion = DistillationLoss(
        criterion,
        teacher_net,
        distillation_type,
        distillation_alpha,
        distillation_tau
    )

    return criterion


class NetWithLoss(nn.Cell):
    """
    NetWithLoss: Only support Network with Classification.
    """

    def __init__(self, model, criterion):
        super(NetWithLoss, self).__init__()
        self.model = model
        self.criterion = criterion

    def construct(self, *inputs, **kwargs):
        data = inputs[0]
        label = inputs[1]
        predict = self.model(data)
        loss = self.criterion(data, predict, label)
        return loss
