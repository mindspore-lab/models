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
import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('PKSD_mindspore')] + 'PKSD_mindspore')  # 这里要改为你自己的项目的主目录

import json
import random
from longling import wf_open
from tqdm import tqdm
from EduSim.utils import get_proj_path

data_path = f'{get_proj_path()}/data/dataProcess/junyi/student_log_kt_None'
with open(data_path, 'r', encoding="utf-8") as f:
    datatxt = f.readlines()

dataOff_path = f'{get_proj_path()}/data/dataProcess/junyi/dataOff'
dataRec_path = f'{get_proj_path()}/data/dataProcess/junyi/dataRec'

random.shuffle(datatxt)
with wf_open(dataOff_path) as wf1, wf_open(dataRec_path) as wf2:
    for i, line in tqdm(enumerate(datatxt), 'splitting...', total=len(datatxt)):
        session = json.loads(line)
        if i <= int(len(datatxt)/2):
            print(json.dumps(session), file=wf1)
        else:
            print(json.dumps(session), file=wf2)
print('data split')
