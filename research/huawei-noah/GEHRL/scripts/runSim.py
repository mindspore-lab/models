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
import logging
import warnings
from argparse import ArgumentParser

import mindspore
from mindspore import context
from EduSim.utils import get_raw_data_path
from EduSim.Envs.KES_ASSIST15 import KESASSISTEnv, kes_assist_train_eval
from EduSim import AbstractAgent

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('GEHRL_mindspore')] + 'GEHRL_mindspore')  # 这里要改为你自己的项目的主目录
os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # 必须放在import各种python的包之前运行

# 忽略warning的输出
warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------
# -s KESassist15 -a HRL -e 99 -r 1 --max_steps 5
# if run HRL, setting:
# --graph_embedding_input              True
# --graph_embedding_type               node2vec
# --HRL_high_policy                    PPO
# --HRL_reward_high_level              env-only
# --HRL_candidates_for_high_level      No
# --HRL_high_goals_encoding            No
# --HRL_subgoals_topo_order_constraint No
# --HRL_deep_high_with__1              No
# --HRL_subgoals_continuity            No
# --HRL_low_policy                     AC
# --HRL_reward_low_level               test_epi
# --HRL_sub_weight                     1.0  # alpha
# --HRL_env_weight                     1.0  # beta
# --HRL_candidaates_for_low_level      embedding  # or "goalprerequisites" for tree-based candidate
# --HRL_low_know_all_goals             disorder
# --HRL_random_low_level               False
# --HRL_asynchronous_train             True
# --HRL_as_tr_episode                  2500
# --HRL_embcan_num                     10


def main():
    context.set_context(device_target='GPU')

    parser = ArgumentParser("learning_path_recommendation")
    simulator = ['KESassist15']
    agent = ['HRL']
    seeds = [1, 5, 10]

    HRL_high_policy = ['PPO', 'AC']
    HRL_reward_high_level = ['dkt', 'test', 'env-only', 'all-dkt', 'all-test', 'rs1', 'rs2', 'rs3']
    HRL_high_goals_encoding = ['No', 'order', 'disorder']
    HRL_subgoals_topo_order_constraint = ['No', 'hard', 'soft']
    HRL_low_policy = ['PPO', 'AC']
    HRL_reward_low_level = ['test_epi', 'dkt', 'test', 'god']
    HRL_candidates_for_low_level = ['embedding', 'No', 'CN', 'goalprerequisites']
    HRL_low_know_all_goals = ['disorder', 'No', 'order', 'disorderTransf']
    HRL_subgoals_continuity = ['No', 'yes']

    # ---------------------------------------------------------- setting models , cuda device at the top of this file
    parser.add_argument('-s', '--simulator', type=str, choices=simulator, default='KESassist15')
    parser.add_argument('-a', '--agent', type=str, choices=agent, default='HRL')
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
    parser.add_argument('--pre_goal', type=int, default=-1)
    parser.add_argument('--cur_goal', type=int, default=-1)
    parser.add_argument('--cur_goal_count', type=int, default=0)
    parser.add_argument('--current_rec_log', type=list, default=[])
    parser.add_argument('--repe_abandon_list', type=list, default=[])
    parser.add_argument('--item_count_dict', type=dict, default={})
    # -------------------------------------------------------------------------------------------- 全局常量
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00005)  # original: 0.00005
    parser.add_argument('-ppoclip', '--ppoclip', type=float, default=0.1)  # original: 0.1
    parser.add_argument('--perfect_log_max_length', type=int, default=50)
    parser.add_argument('--cudaDevice', type=str, default='cuda')
    parser.add_argument('--seed', type=int, choices=seeds)
    parser.add_argument('--max_episode_num', type=int, default=15000)
    parser.add_argument('--dataRecPath', type=str, default='')
    # --------------------------------------------------------------------------------------------HRL
    parser.add_argument('-gonat', '--gonathresh', type=float, default=0.9)
    parser.add_argument('--graph_embedding_input', type=bool, default=True)
    parser.add_argument('--graph_embedding_type', type=str, default='node2vec')
    parser.add_argument('--HRL_high_policy', type=str, choices=HRL_high_policy, default='PPO')
    parser.add_argument('--HRL_reward_high_level', type=str, choices=HRL_reward_high_level,
                        default='env-only')
    parser.add_argument('--HRL_candidates_for_high_level', type=str, default='No')
    parser.add_argument('--HRL_high_goals_encoding', type=str,
                        choices=HRL_high_goals_encoding, default='No')
    parser.add_argument('--HRL_subgoals_topo_order_constraint', type=str,
                        choices=HRL_subgoals_topo_order_constraint, default='No')
    parser.add_argument('--HRL_deep_high_with__1', type=str, default='No')
    parser.add_argument('--HRL_subgoals_continuity', type=str,
                        choices=HRL_subgoals_continuity, default='No')

    parser.add_argument('--HRL_low_policy', type=str, choices=HRL_low_policy, default='AC')
    parser.add_argument('--HRL_reward_low_level', type=str,
                        choices=HRL_reward_low_level, default='test_epi')
    parser.add_argument('--HRL_sub_weight', type=float, default=1.0)
    parser.add_argument('--HRL_env_weight', type=float, default=1.0)
    parser.add_argument('--HRL_candidaates_for_low_level', type=str,
                        choices=HRL_candidates_for_low_level, default='embedding')
    parser.add_argument('--HRL_low_know_all_goals', type=str,
                        choices=HRL_low_know_all_goals, default='disorder')

    parser.add_argument('--HRL_random_low_level', type=bool, default=False)
    parser.add_argument('--HRL_asynchronous_train', type=bool, default=True)
    parser.add_argument('--HRL_as_tr_episode', type=int, default=1)
    parser.add_argument('--HRL_embcan_num', type=int, default=60)
    args = parser.parse_args().__dict__
    args['seed'] = seeds[args['repeat_num'] - 1]
    if args['simulator'] == 'KESassist15':
        args['dataRecPath'] = f'{get_raw_data_path()}/ASSISTments2015/processed/'
        args['ppoclip'] = 0.9
    if args['simulator'] == 'KESassist15':
        env = KESASSISTEnv(dataRec_path=args['dataRecPath'], seed=args['seed'])
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

        kes_assist_train_eval(env_input_dict)


if __name__ == '__main__':
    main()
