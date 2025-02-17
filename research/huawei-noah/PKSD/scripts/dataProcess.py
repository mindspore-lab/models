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
# -*- coding:utf-8 _*-
import subprocess
import sys
import os
from EduSim.utils import get_proj_path, get_raw_data_path
from EduSim.Envs.KES.junyi_process import extract_relations, build_json_sequence
sys.path.append(os.path.dirname(sys.path[0]))  # 这里要改为你自己的项目的主目录
already_log_finish_list = []


def check_finish(tasks, runningtasks):
    for i, task in enumerate(tasks):
        returncode = task.poll()
        if returncode is not None:
            if i not in already_log_finish_list:
                print(f"{runningtasks[i].split()[1]}-{runningtasks[i].split()[5]} finished")
                already_log_finish_list.append(i)


def get_available_devices():
    # 检测可用的device
    available_devices = []
    gpu_status = subprocess.check_output('nvidia-smi', shell=True)
    gpu_status = gpu_status.decode('GBK')
    gpu_status = gpu_status.split('\n')
    i = 0
    for line in gpu_status:
        if 'Default' in line:
            info = line.split()
            mems = []
            cals = []
            for gpu_sta in info:
                if 'MiB' in gpu_sta:
                    mems.append(int(gpu_sta[:-3]))
                if '%' in gpu_sta and '|' not in gpu_sta:
                    cals.append(int(gpu_sta[:-1]))
            used_mem = mems[0]
            # all_mem = mems[1]
            used_mem = used_mem / 1024
            cal_used = cals[-1]
            if cal_used <= 5 and used_mem <= 0.85:
                available_devices.append([i, cal_used])
            i = i + 1
    available_devices = sorted(available_devices, key=lambda x: x[1])  # 从小到大排序
    available_devices = [pair[0] for pair in available_devices]
    return available_devices


if __name__ == '__main__':
    cuda_device = 0
    print('Data process start')
    src_root = f'{get_raw_data_path()}/ASSISTments2015/processed/'
    tar_root = f'{get_proj_path()}/data/dataProcess/ASSISTments2015/'
    os.makedirs(tar_root, exist_ok=True)

    # junyi
    # extract_relations(src_root=src_root, tar_root=tar_root)
    # build_json_sequence(src_root=src_root, tar_root=tar_root, ku_dict_path=tar_root + 'graph_vertex.json', n=None)
    # os.system(f'python {get_proj_path()}/EduSim/Envs/KES/meta/split_data.py')
    # os.system(f'python {get_proj_path()}/EduSim/Envs/KES/BuildTransitionGraph.py')
    # os.system(f'CUDA_VISIBLE_DEVICES={cuda_device} python {get_proj_path()}/EduSim/Envs/KES/envDKT.py')
    # os.system(f'CUDA_VISIBLE_DEVICES={cuda_device} python {get_proj_path()}/EduSim/GraphEmbedding.py 1')

    # assist15
    # os.system(f'python {get_proj_path()}/EduSim/Envs/KES_ASSIST15/BuildGraph.py')
    # os.system(f'CUDA_VISIBLE_DEVICES={cuda_device} python {get_proj_path()}/EduSim/Envs/KES_ASSIST15/envDKT.py')
    # os.system(f'CUDA_VISIBLE_DEVICES={cuda_device} python {get_proj_path()}/EduSim/GraphEmbedding.py 2')

    # assist09
    # os.system(f'python {get_proj_path()}/EduSim/Envs/KES_ASSIST09/BuildGraph.py')
    # os.system(f'CUDA_VISIBLE_DEVICES={cuda_device} python {get_proj_path()}/EduSim/Envs/KES_ASSIST09/envDKT.py')


