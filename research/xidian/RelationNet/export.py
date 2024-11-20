# Copyright 2024 Xidian University
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

"""export checkpoint file into air, mindir models"""
import os
import numpy as np
from omniglot_train_few_shot import CNNEncoder, RelationNetwork
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, export, context
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target)


    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(64, 8)
    encoder_checkpoint = os.path.join(current_dir, config.model_root, config.encoder_checkpoint)
    relation_checkpoint = os.path.join(current_dir, config.model_root, config.relation_checkpoint)

    param_dict_f = ms.load_checkpoint(encoder_checkpoint)
    param_dict_r = ms.load_checkpoint(relation_checkpoint)
    ms.load_param_into_net(feature_encoder, param_dict_f)
    ms.load_param_into_net(relation_network, param_dict_r)
    feature_encoder.set_train()
    relation_network.set_train()

    net1 = feature_encoder
    input_arr1 = Tensor(np.ones(
        [5, 1, 28, 28]), ms.float32)
    export(net1, input_arr1, file_name=config.file_name1,
           file_format=config.file_format)
    
    net2 = relation_network
    input_arr2 = Tensor(np.ones(
        [25, 128, 5, 5]), ms.float32)
    export(net2, input_arr2, file_name=config.file_name2,
           file_format=config.file_format)


if __name__ == '__main__':
    run_export()
