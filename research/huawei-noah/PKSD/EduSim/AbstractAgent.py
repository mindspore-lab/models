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
# coding: utf-8
import copy
import time
import os
import numpy as np

import mindspore
from mindspore import nn
from mindspore.ops import composite as C

from EduSim.agents import PKSD
from EduSim.buffer import ReplayBuffer
from EduSim.utils import get_feature_matrix, compute_dkt_loss, episode_reward_reshape, mds_concat
from EduSim.utils import get_proj_path
from EduSim.deep_model import DKTnet, RnnEncoder


class AbstractAgent:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.max_sequence_length = args['max_steps']
        self.action_space = env.action_space
        self.state_t = []
        self.no_target_state_t = mindspore.Tensor([])
        self.rl_state_t = mindspore.Tensor([])
        self.pre_log = None
        self.tmp_logs = []
        self.learner_profile = []
        self.learner_targets = []
        self.DKT_states = []
        self.episode_start_time = time.time()
        self.episode_end_time = time.time()
        self.pre_step_ks = []
        self.after_step_ks = []

        # 可更改的超参数
        self.action_dim = len(list(env.action_space))
        self.num_skills = self.action_dim
        self.DKT_input_dim = 2 * self.action_dim

        self.RNN_encoder_input_dim = 2 * self.action_dim
        self.RNN_encoder_output_dim = 4 * self.action_dim
        self.RL_input_dim = self.RNN_encoder_output_dim + self.action_dim

        self.learning_rate = self.args['learning_rate']
        self.batch_size = 16
        self.buffer_size = 1310720  # 65536*2 = 2^17
        self.gamma = 0.98  # 折扣因子
        self.epoch_num = 10
        self.target_update = 10  # 目标网络更新频率
        self.use_gpu = True
        self.begin_train_length = 1000
        self.dktnet_trian = False
        self.random_policy = False

        # 实验设置
        self.experiment_idx = args['experiment_idx']
        self.repeat_num = args['repeat_num']
        print(f'Current Experiment Index:{self.experiment_idx}_{self.repeat_num}')
        self.log_dir_name = (f'{get_proj_path()}/EduSim/Experiment_logs/'
                             f'{self.env.env_name}/Experiment_{self.experiment_idx}/')
        os.makedirs(self.log_dir_name, exist_ok=True)

        # DKTnet设置
        if self.env.env_name == 'KESassist15':
            self.embed_dim = 64
            self.DKT_hidden_size = 128
            self.hidden_size_1 = 128
            self.hidden_size_2 = 64
            self.hidden_size_3 = 32
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST15/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KSS':
            self.embed_dim = 15
            self.DKT_hidden_size = 20
            self.hidden_size_1 = 128
            self.hidden_size_2 = 64
            self.hidden_size_3 = 32
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KSS/meta_data/DKT_USE_DATA/all_data_trained_DKT_model.pth'
        dkt_para_dict = {
            'input_size': self.DKT_input_dim,
            'emb_dim': self.embed_dim,
            'hidden_size': self.DKT_hidden_size,
            'num_skills': self.action_dim,
            'nlayers': 2,
            'dropout': 0.0,
        }
        self.DKTnet = DKTnet(dkt_para_dict)
        if os.path.exists(self.DKT_model_path):
            param_dict = mindspore.load_checkpoint(self.DKT_model_path)
            _, _ = mindspore.load_param_into_net(self.DKTnet, param_dict)

        # Agent设置
        self.buffer_size = 1
        self.begin_train_length = 0
        self.epoch_num = 1
        c2 = 0.001
        pksd_para_dict = {
            'input_dim': self.RL_input_dim,
            'output_dim': self.action_dim,
            'hidden_dim1': self.hidden_size_1,
            'hidden_dim2': self.hidden_size_3,
            'env': self.env,
            'lr_policy': self.learning_rate,
            'lr_pec': 0.0001,
            'lr_imp': 0.0001,
            'gamma': self.gamma,
            'c2': c2,
            'RNN_encoder_input_dim': self.RNN_encoder_input_dim,
            'RNN_encoder_output_dim': self.RNN_encoder_output_dim,
            'embed_dim': self.embed_dim,
            'args': self.args,
        }

        self.agent = PKSD(pksd_para_dict)
        self.DKTnet_optimizer = nn.Adam(self.DKTnet.trainable_params(), learning_rate=0.0001)
        self.grad_fn_dkt = mindspore.value_and_grad(self.dkt_forward_fn, None,
                                                    weights=self.DKTnet.trainable_params())

        # Buffer设置
        self.train_buffer = ReplayBuffer(self.buffer_size, episode_length=self.max_sequence_length)

    def begin_episode(self, learner_profile):
        # 一个新的episode的state_0  不用初始learner_profile的版本
        # 全局变量初始化
        self.args['step_count'] = 0
        self.args['steptime_state_saver'] = mindspore.Tensor([], dtype=mindspore.float32)
        self.args['steptime_knowledge_state'] = mindspore.Tensor([], dtype=mindspore.float32)
        self.args['steptime_next_state_saver'] = mindspore.Tensor([], dtype=mindspore.float32)
        self.args['steptime_perfect_log_one_hot'] = mindspore.Tensor([], dtype=mindspore.float32)
        self.args['learner_initial_logs'] = copy.deepcopy(self.env._learner._logs)
        self.args['episode_subgoals'] = []
        self.args['steptime_dkt_ks'] = mindspore.Tensor([], dtype=mindspore.float32)
        self.args['current_rec_log'] = []
        self.args['repe_abandon_list'] = []
        self.args['item_count_dict'] = {}

        self.state_t = []
        self.learner_targets = list(learner_profile[0]['target'])
        self.learner_profile = learner_profile[0]['logs']
        self.rl_state_t = mindspore.Tensor([], dtype=mindspore.float32)
        self.tmp_logs = []
        self.train_buffer.her_scores = []
        self.DKT_states = []
        self.agent.pre_learn_node = '-1'
        self.episode_start_time = time.time()

    def step(self):
        self.pre_step_ks = self.env._learner._state

        one_hot_data = get_feature_matrix(self.state_t,
                                          self.action_dim,
                                          self.RNN_encoder_input_dim)
        one_hot_data = one_hot_data.unsqueeze(0)

        cur_DKT_states = mindspore.ops.sigmoid(self.DKTnet(one_hot_data))[max(0, len(self.state_t) - 1), 0, :]
        self.args['steptime_dkt_ks'] = cur_DKT_states
        candidates = [i for i in range(self.action_dim)]

        # 获得做题记录编码，并输入RL模型
        input_data = get_feature_matrix(self.state_t, self.action_dim, self.RNN_encoder_input_dim)
        input_data = input_data.unsqueeze(0)  # 在0处增加一维，相当于batch_size = 1
        state_input_dict = {
            'states': input_data,
            'states_lengths_ids': mindspore.Tensor([max(0, len(self.state_t) - 1)]),
            'targets': [self.learner_targets]
        }
        agent_step = self.agent.step(state_input_dict, candidates)

        if self.train_buffer.size() * self.max_sequence_length < self.begin_train_length or self.random_policy:
            item = np.random.randint(self.action_dim)
        else:
            item = int(agent_step)
        if self.env.env_name == 'KSS':
            item = str(item)
        return item

    def observe(self, observation, reward, done, info):
        self.after_step_ks = self.env._learner._state

        self.agent.pre_learn_node = observation[0]
        # state_t - > state_t+1
        state = copy.deepcopy(self.state_t)
        # 获取next_state
        self.state_t.append(list(observation))
        self.args['current_rec_log'].append(list(observation))
        next_state = copy.deepcopy(self.state_t)
        self.tmp_logs.append([state, observation[0], reward, next_state, done])

        self.args['step_count'] += 1

    def end_episode(self, observation, reward, done, info):
        # self.tmp_logs[-1][4] = True  # 为了TD计算Qvalue和v（s）正确，需要done在episode结束时为True
        items = []
        answers = []
        for log in self.state_t:
            items.append(str(log[0]))
            answers.append(str(log[1]))
        num = [str(len(items))]
        self.pre_log = (num, items, answers)
        print()
        print(f"episode count: {self.args['episode_count']}")
        print(self.pre_log)
        print('reward: ' + str(reward))
        print('info' + str(info))

        episode_reward_reshape(self.tmp_logs, reward)
        for i, el in enumerate(self.tmp_logs):
            # store
            targets = copy.deepcopy(self.learner_targets)
            el[0] = {'state': copy.deepcopy(el[0]), 'targets': targets}
            el[3] = {'next_state': copy.deepcopy(el[3]), 'targets': targets}
            self.train_buffer.add(el[0], el[1], el[2], el[3], el[4], i)


        file_name = str(self.repeat_num) + '.txt'
        if self.args['episode_count'] == 1:
            writing_mode = 'w'
        else:
            writing_mode = 'a'
        with open(self.log_dir_name + file_name, writing_mode) as f:
            lines = []
            if writing_mode == 'w':
                lines.append(f'{self.args}')
            lines.append(f'{items} {answers} \n')
            lines.append(f'reward:{reward}, target:{self.learner_targets}\n')
            lines.append(f'episode_reward: {reward} \n')

            result = ''
            for line in lines:
                result = result + line
            f.write(result)


        if (self.train_buffer.size() * self.max_sequence_length > self.begin_train_length and
                not self.random_policy):  # 当buffer数据数量超过一定值后，才进行Q网络训练
            for _ in range(self.epoch_num):
                b_s, b_a, b_r, b_ns, b_d = self.train_buffer.sample(self.batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                self.update(transition_dict)

        # 统计episode time
        self.args['episode_count'] += 1
        self.episode_end_time = time.time()
        # print(f'{self.log_dir_name}{self.repeat_num}.txt\n'
        #       f'episode_time:{self.episode_end_time - self.episode_start_time}')

    def dkt_forward_fn(self, batch_data_states, batch_data_one_hot):
        dktnet_states = self.DKTnet(batch_data_states)
        dkt_loss = compute_dkt_loss(dktnet_states, batch_data_one_hot)
        return dkt_loss

    def update(self, transition_dict):
        # 构造输入，并给rnnEncoder
        rewards = mindspore.Tensor(transition_dict['rewards'], dtype=mindspore.float32).view(-1, 1)
        actions = mindspore.Tensor([int(item) for item in transition_dict['actions']]).view(-1, 1)
        dones = mindspore.Tensor(transition_dict['dones'], dtype=mindspore.float32).view(-1, 1)

        # states encoding
        states = mindspore.Tensor([], dtype=mindspore.float32)
        next_states = mindspore.Tensor([], dtype=mindspore.float32)
        for i in range(len(transition_dict['states'])):
            states = mds_concat((states,
                                 get_feature_matrix(transition_dict['states'][i]['state'],
                                                    self.action_dim,
                                                    self.RNN_encoder_input_dim).unsqueeze(0)), 0)
            next_states = mds_concat((next_states,
                                      get_feature_matrix(transition_dict['next_states'][i]['next_state'],
                                                         self.action_dim,
                                                         self.RNN_encoder_input_dim).unsqueeze(0)), 0)
        # DKT train
        if self.dktnet_trian:
            batch_data_states = mindspore.Tensor([], dtype=mindspore.float32)
            for i in range(len(transition_dict['states'])):
                batch_data_states = mds_concat((batch_data_states,
                                                get_feature_matrix(transition_dict['states'][i]['state'],
                                                                   self.action_dim,
                                                                   self.DKT_input_dim).unsqueeze(0)), 0)

            batch_data_one_hot = mindspore.Tensor([], dtype=mindspore.float32)
            for i in range(len(transition_dict['states'])):
                batch_data_one_hot = mds_concat((batch_data_one_hot,
                                                 get_feature_matrix(transition_dict['states'][i]['state'],
                                                                    self.action_dim,
                                                                    self.DKT_input_dim
                                                                    ).unsqueeze(0)), 0)

            _, dkt_grads = self.grad_fn_dkt(batch_data_states, batch_data_one_hot)
            dkt_grads = C.clip_by_value(dkt_grads,
                                        clip_value_max=mindspore.Tensor(self.args['grad_clip'],
                                                                        mindspore.dtype.float32))
            self.DKTnet_optimizer(dkt_grads)

        targets = [item['targets'] for item in transition_dict['states']]
        next_states_lengths_ids = mindspore.Tensor([max(0, len(sequence['next_state']) - 1)
                                                    for sequence in transition_dict['next_states']])
        states_lengths_ids = mindspore.Tensor([max(0, len(sequence['state']) - 1)
                                                    for sequence in transition_dict['states']])

        # 训练
        episode_end_dict = {
            'states': states,
            'next_states': next_states,
            'actions': actions,
            'dones': dones,
            'states_lengths_ids': states_lengths_ids,
            'next_states_lengths_ids': next_states_lengths_ids,
            'targets': targets,
            'rewards': rewards
        }

        critic_loss, actor_loss, regress_loss = self.agent.learn(episode_end_dict)
        print(f"critic_loss: {critic_loss},   actor_loss: {actor_loss},   regress_loss: {regress_loss}")

    def n_step(self, max_steps: int):
        return [self.step() for _ in range(max_steps)]
