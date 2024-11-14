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
"""Resnet."""

from typing import Optional, Type, Union, List

import mindspore.nn as nn

from mindvision.classification.models.backbones import ResidualBlockBase, ResidualBlock, ResNet
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.head import DenseHead
from mindvision.classification.models.neck import GlobalAvgPooling
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

__all__ = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152'
]


def _resnet(arch: str,
            block: Type[Union[ResidualBlockBase, ResidualBlock]],
            layers: List[int],
            num_classes: int,
            pretrained: bool,
            input_channel: int,
            group: int = 1,
            base_width: int = 64,
            norm: Optional[nn.Cell] = None
            ) -> ResNet:
    """ResNet architecture."""
    backbone = ResNet(
        block=block,
        layer_nums=layers,
        group=group,
        base_width=base_width,
        norm=norm
    )
    neck = GlobalAvgPooling()
    head = DenseHead(input_channel=input_channel, num_classes=num_classes)
    model = BaseClassifier(backbone, neck, head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model


def resnet18(num_classes: int = 1000,
             pretrained: bool = False,
             group: int = 1,
             base_width: int = 64,
             norm: Optional[nn.Cell] = None
             ) -> ResNet:
    """
    Constructs a ResNet-18 architecture from
    `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        num_classes (int): The number of classification. Default: 1000.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        group (int): The number of group convolutions. Default: 1.
        base_width (int): The width of per group. Default: 64.
        norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import resnet18
        >>>
        >>> net = resnet18()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About ResNet:

    The ResNet is to ease the training of networks that are substantially deeper than those used previously.
    The model explicitly reformulate the layers as learning residual functions with reference to the layer inputs,
    instead of learning unreferenced functions.

    Citation:

    .. code-block::

        @article{2016Deep,
        title={Deep Residual Learning for Image Recognition},
        author={ He, K.  and  Zhang, X.  and  Ren, S.  and  Sun, J. },
        journal={IEEE},
        year={2016},
        }
    """
    return _resnet(
        "resnet18", ResidualBlockBase, [
            2, 2, 2, 2], num_classes, pretrained, 512, group, base_width, norm)


def resnet34(num_classes: int = 1000,
             pretrained: bool = False,
             group: int = 1,
             base_width: int = 64,
             norm: Optional[nn.Cell] = None
             ) -> ResNet:
    """
    Constructs a ResNet-34 architecture from
    `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        num_classes (int): The number of classification. Default: 1000.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        group (int): The number of group convolutions. Default: 1.
        base_width (int): The width of per group. Default: 64.
        norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import resnet34
        >>>
        >>> net = resnet34()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About ResNet:

    The ResNet is to ease the training of networks that are substantially deeper than those used previously.
    The model explicitly reformulate the layers as learning residual functions with reference to the layer inputs,
    instead of learning unreferenced functions.

    Citation:

    .. code-block::

        @article{2016Deep,
        title={Deep Residual Learning for Image Recognition},
        author={ He, K.  and  Zhang, X.  and  Ren, S.  and  Sun, J. },
        journal={IEEE},
        year={2016},
        }
    """
    return _resnet(
        "resnet34", ResidualBlockBase, [
            3, 4, 6, 3], num_classes, pretrained, 512, group, base_width, norm)


def resnet50(num_classes: int = 1000,
             pretrained: bool = False,
             group: int = 1,
             base_width: int = 64,
             norm: Optional[nn.Cell] = None
             ) -> ResNet:
    """
    Constructs a ResNet-50 architecture from
    `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        num_classes (int): The number of classification. Default: 1000.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        group (int): The number of group convolutions. Default: 1.
        base_width (int): The width of per group. Default: 64.
        norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import resnet50
        >>>
        >>> net = resnet50()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About ResNet:

    The ResNet is to ease the training of networks that are substantially deeper than those used previously.
    The model explicitly reformulate the layers as learning residual functions with reference to the layer inputs,
    instead of learning unreferenced functions.

    Citation:

    .. code-block::

        @article{2016Deep,
        title={Deep Residual Learning for Image Recognition},
        author={ He, K.  and  Zhang, X.  and  Ren, S.  and  Sun, J. },
        journal={IEEE},
        year={2016},
        }
    """
    return _resnet(
        "resnet50", ResidualBlock, [
            3, 4, 6, 3], num_classes, pretrained, 2048, group, base_width, norm)


def resnet101(num_classes: int = 1000,
              pretrained: bool = False,
              group: int = 1,
              base_width: int = 64,
              norm: Optional[nn.Cell] = None
              ) -> ResNet:
    """
    Constructs a ResNet-101 architecture from
    `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        num_classes (int): The number of classification. Default: 1000.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        group (int): The number of group convolutions. Default: 1.
        base_width (int): The width of per group. Default: 64.
        norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import resnet101
        >>>
        >>> net = resnet101()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About ResNet:

    The ResNet is to ease the training of networks that are substantially deeper than those used previously.
    The model explicitly reformulate the layers as learning residual functions with reference to the layer inputs,
    instead of learning unreferenced functions.

    Citation:

    .. code-block::

        @article{2016Deep,
        title={Deep Residual Learning for Image Recognition},
        author={ He, K.  and  Zhang, X.  and  Ren, S.  and  Sun, J. },
        journal={IEEE},
        year={2016},
        }
    """
    return _resnet(
        "resnet101", ResidualBlock, [
            3, 4, 23, 3], num_classes, pretrained, 2048, group, base_width, norm)


def resnet152(num_classes: int = 1000,
              pretrained: bool = False,
              group: int = 1,
              base_width: int = 64,
              norm: Optional[nn.Cell] = None
              ) -> ResNet:
    """
    Constructs a ResNet-152 architecture from
    `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        num_classes (int): The number of classification. Default: 1000.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        group (int): The number of group convolutions. Default: 1.
        base_width (int): The width of per group. Default: 64.
        norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import resnet152
        >>>
        >>> net = resnet152()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About ResNet:

    The ResNet is to ease the training of networks that are substantially deeper than those used previously.
    The model explicitly reformulate the layers as learning residual functions with reference to the layer inputs,
    instead of learning unreferenced functions.

    Citation:

    .. code-block::

        @article{2016Deep,
        title={Deep Residual Learning for Image Recognition},
        author={ He, K.  and  Zhang, X.  and  Ren, S.  and  Sun, J. },
        journal={IEEE},
        year={2016},
        }
    """
    return _resnet(
        "resnet152", ResidualBlock, [
            3, 8, 36, 3], num_classes, pretrained, 2048, group, base_width, norm)
