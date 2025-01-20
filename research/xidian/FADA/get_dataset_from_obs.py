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

# mox.file.copy_parallel(src_url="./results/src_r101_try_d6/", dst_url='obs://www/code/FADA/ckpt0811/')

# mox.file.copy_parallel(src_url="./", dst_url='obs://www/code/FADA_Final/')

# mox.file.copy_parallel(src_url="obs://www/code/FADA/resnet101-5d3b4d8f.ckpt", dst_url='./pretrained/resnet101-5d3b4d8f.ckpt')


import os

lst = os.listdir('./')
# print(lst)

obs_base_path = 'obs://www/code/FADA_Final_d1/'
save_base_path = './'

for save_name in lst:
    
    if save_name != 'datasets' and save_name != 'results':
        
        save_path = os.path.join(save_base_path, save_name)
        obs_path = os.path.join(obs_base_path, save_name)
        # print(save_path)
        # print(obs_path)
        mox.file.copy_parallel(src_url=save_path, dst_url=obs_path)
