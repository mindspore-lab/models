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

from mindspore.train.serialization import load_checkpoint

# ./pretrained/resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78.ckpt
# moving_mean
# ./pretrained/resnet101-5d3b4d8f.ckpt
param_dict = load_checkpoint('./results/adv_test_d8/model_iter038000_feature_extractor.ckpt')
trans_param_dict = {}
for key, val in param_dict.items():
    if 'bn1' in key and 'moving_mean' in key:
        print(key, val.asnumpy())
        break
        
        

param_dict = load_checkpoint('./results/src_r101_try_d6/model_iter018000_feature_extractor.ckpt')
trans_param_dict = {}
for key, val in param_dict.items():
    if 'bn1' in key and 'moving_mean' in key:
        print(key, val.asnumpy())
        break
        

    