# Copyright 2020 Huawei Technologies Co., Ltd
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
import argparse


def bash_shell(bash_command):
    """
    python 中执行 bash 命令
    :param bash_command:
    :return: bash 命令执行后的控制台输出
    """
    try:
        return os.popen(bash_command).read().strip()
    except:
        return None


def get_config():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument('--cmd', type=str, default='bash run_train.sh')
    return parser.parse_args()


if __name__ == '__main__':
    run_sh_name = get_config().cmd
    if not run_sh_name.endswith('.sh'):
        raise ValueError('The run file is error, it\'s name should end with .sh but your input file is {}'.format(run_sh_name))
    print('Run Script:\t{}'.format(run_sh_name))

    print('The Start Path:\t{}'.format(os.getcwd()))
    if os.path.split(os.getcwd())[-1] == 'modelarts':
        os.chdir('../')
        # print(os.getcwd())
    work_path = os.path.join(os.getcwd(), 'scripts')
    os.chdir(work_path)
    print('work_path:\t{}'.format(os.getcwd()))
    os.system(run_sh_name)
