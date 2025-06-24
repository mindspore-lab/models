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
import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('PKSD_mindspore')] + 'PKSD_mindspore')  # 这里要改为你自己的项目的主目录
import logging
import warnings
from argparse import ArgumentParser

import mindspore
from mindspore import context
from EduSim.utils import get_raw_data_path
from EduSim.Envs.KES_ASSIST15 import KESASSISTEnv, kes_assist_train_eval
from EduSim.Envs.KSS import KSSEnv, kss_train_eval
from EduSim import AbstractAgent

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 必须放在import各种python的包之前运行

# 忽略warning的输出
warnings.filterwarnings('ignore')

def main():
    context.set_context(device_target='GPU')

    parser = ArgumentParser("learning_path_recommendation")
    simulator = ['KESassist15', 'KSS']
    agent = ['PKSD']
    seeds = [1, 5, 10]

    # ---------------------------------------------------------- setting models , cuda device at the top of this file
    parser.add_argument('-s', '--simulator', type=str, choices=simulator, default='KSS')
    parser.add_argument('-a', '--agent', type=str, choices=agent, default='PKSD')
    parser.add_argument('-e', '--experiment_idx', type=int, default=2, help='Experiment id for one model')
    parser.add_argument('-r', '--repeat_num', type=int, default=1, help='Experiment id for one seed')
    parser.add_argument('-m', '--max_steps', type=int, default=20)
    parser.add_argument('-k', '--know_all_log', type=bool, default=False)
    parser.add_argument('-RC', '--Repe_control', type=bool, default=False)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    # -------------------------------------------------------------------------------------------- 全局变量
    parser.add_argument('--step_count', type=int, default=0)
    parser.add_argument('--episode_count', type=int, default=0)
    parser.add_argument('--steptime_state_saver', default=mindspore.Tensor([],
                                                                           dtype=mindspore.float32))
    parser.add_argument('--steptime_next_state_saver', default=mindspore.Tensor([],
                                                                                dtype=mindspore.float32))
    parser.add_argument('--steptime_perfect_log_one_hot', default=mindspore.Tensor([],
                                                                                   dtype=mindspore.float32))
    parser.add_argument('--steptime_knowledge_state', default=mindspore.Tensor([],
                                                                               dtype=mindspore.float32))
    parser.add_argument('--learner_initial_logs', default=[])
    parser.add_argument('--steptime_dkt_ks', default=mindspore.Tensor([],
                                                                      dtype=mindspore.float32))
    parser.add_argument('--episode_subgoals', default=[])
    parser.add_argument('--current_rec_log', type=list, default=[])
    parser.add_argument('--repe_abandon_list', type=list, default=[])
    parser.add_argument('--item_count_dict', type=dict, default={})
    # -------------------------------------------------------------------------------------------- 全局常量
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00005)  # original: 0.00005
    parser.add_argument('-ppoclip', '--ppoclip', type=float, default=0.1)  # original: 0.1
    parser.add_argument('--perfect_log_max_length', type=int, default=50)
    parser.add_argument('--cudaDevice', type=str, default='cuda')
    parser.add_argument('--seed', type=int, choices=seeds)
    parser.add_argument('--max_episode_num', type=int, default=30000)
    parser.add_argument('--dataRecPath', type=str, default='')
    # --------------------------------------------------------------------------------------------PKSD
    parser.add_argument('--PKSD_base_policy', type=str, choices=['AC'], default='AC')
    parser.add_argument('-ppe', '--PKSD_PerEncoder', type=str, choices=['MLP', 'GNN'], default='MLP')
    parser.add_argument('-pie', '--PKSD_ImPerEncoder', type=str, choices=['RNN', 'RNN_GNN'], default='RNN')
    parser.add_argument('-p1er', '--PKSD_phase_1_ratio', type=float, default=0.3)

    args = parser.parse_args().__dict__
    args['seed'] = seeds[args['repeat_num'] - 1]

    if 'GNN' in args['PKSD_PerEncoder']:
        args['PKSD_ImPerEncoder'] = 'RNN_GNN'

    if args['simulator'] == 'KESassist15':
        args['dataRecPath'] = f'{get_raw_data_path()}/ASSISTments2015/processed/'
        args['ppoclip'] = 0.9
        env = KESASSISTEnv(dataRec_path=args['dataRecPath'], seed=args['seed'])
    elif args['simulator'] == 'KSS':
        env = KSSEnv(seed=args['seed'])
        args['ppoclip'] = 0.9

    agent = AbstractAgent(env, args)
    env_input_dict = {}
    env_input_dict['agent'] = agent
    env_input_dict['env'] = env
    env_input_dict['max_steps'] = args['max_steps']
    env_input_dict['max_episode_num'] = args['max_episode_num']
    env_input_dict['level'] = "summary"
    env_input_dict['n_step'] = False
    env_input_dict['train'] = False
    env_input_dict['logger'] = logging
    env_input_dict['values'] = None
    env_input_dict['monitor'] = None

    if args['simulator'] == 'KESassist15':
        kes_assist_train_eval(env_input_dict)
    elif args['simulator'] == 'KSS':
        kss_train_eval(env_input_dict)


if __name__ == '__main__':
    main()
