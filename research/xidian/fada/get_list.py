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

import os


# img_path = '../../datasets/gtav/images/'
# img_list = os.listdir(img_path)
# print('img_list: ', img_list)
#
#
# with open('../../datasets/gtav/gta5_train_list.txt', 'w') as f:
#     for img_name in img_list:
#         f.write(img_name + '\n')

img_path = '../../datasets/cityscape/leftImg8bit/val'
img_list = os.listdir(img_path)
print('img_list: ', img_list)

res = []
for root, dirs, files in os.walk(img_path):
    print('root_dir:', root)
    print('sub_dirs:', dirs)
    print('files:', files)
    pre = root.split('/')[-1]
    res += [pre + '/' + item for item in files]

print(res)

with open('../../datasets/cityscape/cityscapes_val_list.txt', 'w') as f:
    for img_name in res:
        f.write(img_name + '\n')
