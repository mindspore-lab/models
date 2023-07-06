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
"""
##############export checkpoint file into air, mindir models#################
python export.py
"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.models import Backbone,Class_classifier
from model_utils.config import config

def main():
    backbone = Backbone()
    classifier = Class_classifier()
    backbone_param_dict = load_checkpoint(config.backbone_ckpt_file)
    classifier_param_dict = load_checkpoint(config.classifier_ckpt_file)

    load_param_into_net(backbone, backbone_param_dict)
    load_param_into_net(classifier, classifier_param_dict)

    input_arr = Tensor(np.ones([1, 1, config.imageSize, config.imageSize]), ms.float32)

    net = ms.nn.SequentialCell([backbone, classifier])
    export(net, input_arr, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    main()

