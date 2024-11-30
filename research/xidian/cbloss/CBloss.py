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

"""CBloss"""
import numpy as np
import mindspore as ms
import mindspore.ops as ops


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = ops.binary_cross_entropy_with_logits(logits=logits, label=labels, reduction="mean")
    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = ops.exp(-gamma * labels * logits - gamma * ops.log(1 + ops.exp(-1.0 * logits)))
    loss = modulator * BCLoss
    weighted_loss = alpha * loss
    focal_loss = ops.sum(weighted_loss)
    # Normalize by the total number of positive samples.
    focal_loss /= ops.sum(labels)
    return focal_loss

def CB_loss(labels, logits, samples_per_cls, num_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      num_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      class_balanced_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_of_classes
    onehot = ops.OneHot()
    depth, on_value, off_value = 10, ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32)
    labels_one_hot = onehot(labels, depth, on_value, off_value).float()
    weights = ms.Tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.tile((labels_one_hot.shape[0], 1)) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.tile((1, num_of_classes))
    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = ops.binary_cross_entropy_with_logits(logits, labels_one_hot, weights)
    elif loss_type == "softmax":
        pred = ops.softmax(logits,axis=1)
        cb_loss = ops.binary_cross_entropy(pred, labels_one_hot, weights)
    else:
        raise ValueError("Please input correct loss_type!")
    return cb_loss


if __name__ == '__main__':
    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend",device_id =5)#PYNATIVE_MODE
    outputs = ms.Tensor([[1,2,3,4,5,6,7,8,9,10],[2,4,6,8,10,1,3,5,7,9]]).astype('float32')
    # outputs = ops.randn((2, 10))
    labels = ops.ones(2).astype('int32')
    num_per_classes = np.array(
        [int(np.floor(5000 * ((1 / 100) ** (1 / 9)) ** (i))) for i in range(10)])
    loss = CB_loss(labels=labels, logits=outputs,
                   samples_per_cls=num_per_classes, num_of_classes=10,
                   loss_type="softmax", beta=0.9999, gamma=2)
    print(loss)