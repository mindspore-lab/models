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

import moxing as mox

# mox.file.copy_parallel(src_url="obs://www/datasets/cityspace/", dst_url='./datasets/cityspace')

# mox.file.copy_parallel(src_url="./results/src_r101_try/", dst_url='obs://www/code/FADA/ckpt0730/')

mox.file.copy_parallel(src_url="obs://www/code/advseg/premodel/DeepLab_resnet_pretrained_init-f81d91e8.ckpt", dst_url='./pretrained/DeepLab_resnet_pretrained_imagenet_init.ckpt')

