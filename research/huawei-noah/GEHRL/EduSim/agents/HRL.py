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
import random
import numpy as np
import mindspore
from mindspore import nn
from tqdm import tqdm
from EduSim.utils import get_proj_path, mds_concat
from .PPO import PPO
from .AC import ActorCritic


class HRL(nn.Cell):
    def __init__(self, hrl_para_dict):
        super().__init__()
        input_dim = hrl_para_dict['input_dim']
        output_dim = hrl_para_dict['output_dim']
        hidden_dim1 = hrl_para_dict['hidden_dim1']
        hidden_dim2 = hrl_para_dict['hidden_dim2']
        env = hrl_para_dict['env']
        lr_big = hrl_para_dict['lr_big']
        lr_lit = hrl_para_dict['lr_lit']
        gamma = hrl_para_dict['gamma']
        c2 = hrl_para_dict['c2']
        args = hrl_para_dict['args']
        outer_encoder = hrl_para_dict['outer_encoder']
        RNN_encoder_output_dim = hrl_para_dict['RNN_encoder_output_dim']

        self.name = 'HRL'
        self.policy_mode = 'on_policy'
        self.pre_learn_node = '-1'
        self.gamma = gamma
        self.lr_big = lr_big
        self.lr_lit = lr_lit
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.env = env
        self.c2 = c2
        self.episode_count = 0
        self.args = args
        self.big_episode_reward_DKT_states = []
        self.RNN_encoder_output_dim = RNN_encoder_output_dim

        self.model_settings = {
            'graph_embedding_input': args['graph_embedding_input'],  # graph embedding or not
            'graph_embedding_type': args['graph_embedding_type'],  # node2vec, GCN, GAT, TransE

            'policy high level': args['HRL_high_policy'],  # PPO, AC, Shallow
            'reward high level': args['HRL_reward_high_level'],  # dkt, test, env-only, all-dkt, all-test, rs1, rs2, rs3
            'candidates for high level': args['HRL_candidates_for_high_level'],  # No, hard, soft
            'high goals encoding': args['HRL_high_goals_encoding'],  # No, order(transformer), disorder(transformer)
            'subgoals topo order constraint': args['HRL_subgoals_topo_order_constraint'],  # No, hard, soft
            'deep high with -1': args['HRL_deep_high_with__1'],  # No, yes
            'subgoals continuity': args['HRL_subgoals_continuity'],  # No, yes

            'policy low level': args['HRL_low_policy'],  # PP0, AC
            'reward low level': args['HRL_reward_low_level'],  # dkt, test, god, test_epi
            'sub_weight': args['HRL_sub_weight'],
            'env_weight': args['HRL_env_weight'],
            'candidates for low level': args['HRL_candidaates_for_low_level'],  # No, CN, embedding, goalprerequisites
            'low know all goals': args['HRL_low_know_all_goals'],  # No, disorder, order(RNN or Transf), disorderTransf

            'random low level': args['HRL_random_low_level'],
            'asynchronous training': args['HRL_asynchronous_train'],
            'as_tr_episode': args['HRL_as_tr_episode']
        }
        # embedding constrain num
        self.emb_cons_num = args['HRL_embcan_num']
        # graph embedding model
        self.env_graph = self.env.knowledge_structure
        self.env_graph_edges = mindspore.Tensor(list(self.env_graph.edges), dtype=mindspore.int64).t()
        self.initial_graph_embeddings = mindspore.Tensor(self.env.graph_embeddings)
        self.graph_embeddings = self.initial_graph_embeddings
        if self.model_settings['graph_embedding_type'] in ['node2vec']:
            self.graph_embeddings = self.initial_graph_embeddings
        else:
            raise ValueError('graph embedding type not set right!')

        # high goals encoding
        self.high_policy_input_dim = input_dim

        # low know all goals
        self.low_policy_input_dim = self.RNN_encoder_output_dim + self.graph_embeddings.shape[-1] + self.output_dim
        #  some settings
        self.sub_episode_running_steps = 0
        self.sub_episode_running = False
        self.sub_episode = [mindspore.Tensor([])] * 5
        self.all_sub_episodes = []
        self.big_episode = [mindspore.Tensor([])] * 5

        self.sub_episode_max_length = max(int(args['max_steps'] / 4), 3)
        self.cur_sub_goal = -1
        self.left_targets = []
        self.learner_targets = []
        self.sub_intinsic_rewards = mindspore.Tensor([])
        self.order_goals_embed = None
        self.__help_init__(output_dim, hidden_dim1, hidden_dim2, outer_encoder)

    def __help_init__(self, output_dim, hidden_dim1, hidden_dim2, outer_encoder):
        # high leve policy
        if self.model_settings['policy high level'] == 'PPO':
            ppo_para_dict = {
                'input_dim': self.high_policy_input_dim,
                'output_dim': output_dim,
                'hidden_dim1': hidden_dim1,
                'hidden_dim2': hidden_dim2,
                'env': self.env,
                'lr_rate': self.lr_big,
                'gamma': self.gamma,
                'c2': self.c2,
                'args': self.args,
                'outer_encoder': outer_encoder,
                'RLInputCompo': 'RNNwithTarget',
                'policy_clip': self.args['ppoclip'],
                'gae_lambda': 0.95,
                'k_epochs': 5,
                'mini_batch_size': 1,
            }
            self.meta_controller = PPO(ppo_para_dict)
        elif self.model_settings['policy high level'] == 'AC':
            ac_para_dict = {
                'input_dim': self.high_policy_input_dim,
                'output_dim': output_dim,
                'hidden_dim1': hidden_dim1,
                'hidden_dim2': hidden_dim2,
                'env': self.env,
                'lr_rate': self.lr_lit,
                'gamma': self.gamma,
                'args': self.args,
                'outer_encoder': outer_encoder,
                'RLInputCompo': 'RNNwithTarget',
                'lr_sche': False,
            }

            self.meta_controller = ActorCritic(ac_para_dict)
        else:
            raise ValueError('Wrong high level policy setting')

        # low level policy
        if self.model_settings['policy low level'] == 'PPO':
            ppo_para_dict = {
                'input_dim': self.low_policy_input_dim,
                'output_dim': output_dim,
                'hidden_dim1': hidden_dim1,
                'hidden_dim2': hidden_dim2,
                'env': self.env,
                'lr_rate': self.lr_lit,
                'gamma': self.gamma,
                'c2': self.c2,
                'args': self.args,
                'outer_encoder': outer_encoder,
                'RLInputCompo': 'RNNSubgoalTarget',
                'policy_clip': self.args['ppoclip'],
                'gae_lambda': 0.95,
                'k_epochs': 5,
                'mini_batch_size': 1,
            }

            self.controller = PPO(ppo_para_dict)
        elif self.model_settings['policy low level'] == 'AC':
            ac_para_dict = {
                'input_dim': self.low_policy_input_dim,
                'output_dim': output_dim,
                'hidden_dim1': hidden_dim1,
                'hidden_dim2': hidden_dim2,
                'env': self.env,
                'lr_rate': self.lr_lit,
                'gamma': self.gamma,
                'args': self.args,
                'outer_encoder': outer_encoder,
                'RLInputCompo': 'RNNSubgoalTarget',
                'lr_sche': False,
            }

            self.controller = ActorCritic(ac_para_dict)
        else:
            raise ValueError('Wrong low level policy setting')
        # embedding distances
        if self.model_settings['candidates for low level'] == 'embedding':
            EB_CAN_file_path = \
                f'{get_proj_path()}/data/dataProcess/{self.env.env_name}_HRLEBCAN_{self.emb_cons_num}.npy'
            if os.path.exists(EB_CAN_file_path):
                self.embedding_candidates = np.load(EB_CAN_file_path)
            else:
                distances_matrix = mindspore.ops.zeros((self.graph_embeddings.shape[0], self.graph_embeddings.shape[0]))
                for i in tqdm(range(self.graph_embeddings.shape[0])):
                    for j in range(i, self.graph_embeddings.shape[0]):
                        distances_matrix[i, j] = mindspore.ops.norm(self.graph_embeddings[i] - self.graph_embeddings[j])
                self.embedding_candidates = [[]] * self.graph_embeddings.shape[0]
                for node in tqdm(range(self.graph_embeddings.shape[0])):
                    self.embedding_candidates[node] = [[i, distances_matrix[min(node, i), max(node, i)]]
                                                       for i in range(self.graph_embeddings.shape[0])]
                    self.embedding_candidates[node] = sorted(self.embedding_candidates[node], key=lambda x: x[1])
                    self.embedding_candidates[node] = \
                        [pair[0] for i, pair in enumerate(self.embedding_candidates[node]) if i < self.emb_cons_num]

                self.embedding_candidates = np.array(self.embedding_candidates)
                np.save(EB_CAN_file_path, self.embedding_candidates)
            self.embedding_candidates = list(self.embedding_candidates)

    def step(self, states_dict, _):
        # initialization
        if self.args['step_count'] == 0:
            self.sub_episode_running = False
            self.sub_episode_running_steps = 0
            self.sub_episode = [{'states': mindspore.Tensor([]),
                                 'states_lengths_ids': mindspore.Tensor([]),
                                 'targets': [],
                                 'RNN_concate_vector': mindspore.Tensor([])},
                                mindspore.Tensor([]),
                                {'states': mindspore.Tensor([]),
                                 'states_lengths_ids': mindspore.Tensor([]),
                                 'targets': [],
                                 'RNN_concate_vector': mindspore.Tensor([])},
                                mindspore.Tensor([]),
                                mindspore.Tensor([]),
                                ]
            self.all_sub_episodes = []
            self.big_episode = [{'states': mindspore.Tensor([]),
                                 'states_lengths_ids': mindspore.Tensor([]),
                                 'targets': []},
                                mindspore.Tensor([]),
                                {'states': mindspore.Tensor([]),
                                 'states_lengths_ids': mindspore.Tensor([]),
                                 'targets': []},
                                mindspore.Tensor([]),
                                mindspore.Tensor([]),
                                ]
            self.sub_intinsic_rewards = mindspore.Tensor([])
            self.cur_sub_goal = -1
            self.left_targets = []
            self.learner_targets = list(self.env._learner.target)
            self.big_episode_reward_DKT_states = []
            self.order_goals_embed = None

        # sub episode begin
        if not self.sub_episode_running:
            if (self.model_settings['asynchronous training'] and
                    self.episode_count < self.model_settings['as_tr_episode']):
                self.cur_sub_goal = random.choice([i for i in range(len(list(self.env.action_space)))])
            else:
                h_candidates = [i for i in range(len(list(self.env.action_space)))]
                self.cur_sub_goal = self.meta_controller.step(states_dict, candidates=h_candidates)

            if self.big_episode[0]['states'].shape[0] != 0:
                # 错一位添加big_episode的next state
                self.big_episode[2]['states'] = mds_concat(
                    (self.big_episode[2]['states'], states_dict['states']), 0)
                self.big_episode[2]['states_lengths_ids'] = mds_concat(
                    (self.big_episode[2]['states_lengths_ids'], states_dict['states_lengths_ids']), 0)
                self.big_episode[2]['targets'].append(states_dict['targets'][0])

            self.big_episode[0]['states'] = mds_concat(
                (self.big_episode[0]['states'], states_dict['states']), 0)
            self.big_episode[0]['states_lengths_ids'] = mds_concat(
                (self.big_episode[0]['states_lengths_ids'], states_dict['states_lengths_ids']), 0)
            self.big_episode[0]['targets'].append(states_dict['targets'][0])
            self.big_episode[1] = mds_concat(
                (self.big_episode[1], mindspore.Tensor(self.cur_sub_goal).view(1, -1)), 0)

            self.sub_episode = [{'states': mindspore.Tensor([]),
                                 'states_lengths_ids': mindspore.Tensor([]),
                                 'targets': [],
                                 'RNN_concate_vector': mindspore.Tensor([])},
                                mindspore.Tensor([]),
                                {'states': mindspore.Tensor([]),
                                 'states_lengths_ids': mindspore.Tensor([]),
                                 'targets': [],
                                 'RNN_concate_vector': mindspore.Tensor([])},
                                mindspore.Tensor([]),
                                mindspore.Tensor([]),
                                ]
            self.sub_episode_running = True
            self.sub_episode_running_steps = 0

        # controller input
        subgoal_embed = self.graph_embeddings[self.cur_sub_goal].view(1, -1)
        # candidates for low level
        if self.model_settings['candidates for low level'] == 'No':
            low_policy_candidates = [i for i in range(len(list(self.env.action_space)))]
        elif self.model_settings['candidates for low level'] == 'embedding':
            low_policy_candidates = self.get_embedding_candidates()
        else:
            raise ValueError('candidates for low level setting wrong!')

        # low know all goals
        # controller_states = mindspore.ops.cat((controller_states, all_goals_hot), -1)
        controller_states_dict = {
            'states': states_dict['states'],
            'states_lengths_ids': states_dict['states_lengths_ids'],
            'targets': states_dict['targets'],
            'RNN_concate_vector': subgoal_embed
        }
        action = self.controller.step(controller_states_dict, candidates=low_policy_candidates)

        # sub_episode存储
        self.sub_episode[0]['states'] = mds_concat((self.sub_episode[0]['states'],
                                                    controller_states_dict['states']), 0)
        self.sub_episode[0]['states_lengths_ids'] = mds_concat((self.sub_episode[0]['states_lengths_ids'],
                                                                controller_states_dict['states_lengths_ids']), 0)
        self.sub_episode[0]['targets'].append(controller_states_dict['targets'][0])
        self.sub_episode[0]['RNN_concate_vector'] = mds_concat((self.sub_episode[0]['RNN_concate_vector'],
                                                                controller_states_dict['RNN_concate_vector']), 0)
        self.sub_episode[1] = mds_concat((self.sub_episode[1],
                                          mindspore.Tensor(action).view(1, -1)), 0)

        # storage for training
        self.args['episode_subgoals'].append(self.cur_sub_goal)
        if self.model_settings['reward low level'] in ['test_epi', 'test']:
            if self.sub_episode_running_steps == self.sub_episode_max_length - 1:
                action = self.cur_sub_goal
                a = mindspore.Tensor([action])
                self.sub_episode[1][-1] = a
        return action

    def learn(self, RL_next_states_dict, rewards):
        # low level policy train
        start_id = 0
        loss = mindspore.Tensor([0.0])
        for i, sub_episode_data in enumerate(self.all_sub_episodes):
            if self.model_settings['reward low level'] == 'test_epi':
                sub_episode_data[3][-1] = (self.model_settings['sub_weight'] * sub_episode_data[3][-1] +
                                           self.model_settings['env_weight'] * rewards[-1])
                if int(self.big_episode[1][i].asnumpy()) == self.output_dim:  # -1 reward
                    sub_episode_data[3][-1] = sub_episode_data[3][-1] + rewards[-1]
            if self.controller.name == 'ppo':
                epi_length = sub_episode_data[0].shape[0]
                multiepi_sample_ids = [start_id, start_id + epi_length]
                multiepi_done = bool(i == len(self.all_sub_episodes) - 1)
                ac_learn_dict = {
                    'RL_states_dict': sub_episode_data[0],
                    'actions': sub_episode_data[1],
                    'RL_next_states_dict': sub_episode_data[2],
                    'rewards': sub_episode_data[3],
                    'dones': sub_episode_data[4]
                }
                controller_loss = self.controller.learn(ac_learn_dict, multiepi_sample_ids, multiepi_done)
                start_id = start_id + epi_length
                loss = loss + controller_loss
            else:
                ac_learn_dict = {
                    'RL_states_dict': sub_episode_data[0],
                    'actions': sub_episode_data[1],
                    'RL_next_states_dict': sub_episode_data[2],
                    'rewards': sub_episode_data[3],
                    'dones': sub_episode_data[4]
                }
                controller_loss = self.controller.learn(ac_learn_dict)
                loss = loss + controller_loss

        # high level policy train
        if self.model_settings['asynchronous training'] and self.episode_count < self.model_settings['as_tr_episode']:
            if self.meta_controller.name == 'ppo':
                self.meta_controller.ep_old_logprobs = []
                self.meta_controller.continuous_normal_sample = mindspore.Tensor([])
        else:
            # big episode data rebuild
            self.big_episode[3][-1] = rewards[-1]
            final_next_state = RL_next_states_dict['states'][-1].unsqueeze(0).copy()
            final_states_lengths_id = RL_next_states_dict['states_lengths_ids'][-1].view(-1).copy()
            self.big_episode[2]['states'] = mds_concat(
                (self.big_episode[2]['states'], final_next_state), 0)
            self.big_episode[2]['states_lengths_ids'] = mds_concat((self.big_episode[2]['states_lengths_ids'],
                                                                    final_states_lengths_id), 0)
            self.big_episode[2]['targets'].append(RL_next_states_dict['targets'][-1])

            ac_para_dict = {
                'RL_states_dict': self.big_episode[0],
                'actions': self.big_episode[1],
                'RL_next_states_dict': self.big_episode[2],
                'rewards': self.big_episode[3],
                'dones': self.big_episode[4]
            }
            meta_controller_loss = self.meta_controller.learn(ac_para_dict)
            loss = loss + meta_controller_loss

        # episode end
        self.episode_count = self.episode_count + 1
        return loss

    def observe(self, observe_dict):
        observation = observe_dict['observation']
        done = observe_dict['done']
        input_data = observe_dict['input_data']
        states_lengths_ids = observe_dict['states_lengths_ids']
        targets = observe_dict['targets']

        if self.args['step_count'] == 0:
            self.big_episode_reward_DKT_states = self.args['steptime_dkt_ks']

        if self.model_settings['reward low level'] in ['test', 'test_epi']:
            if self.args['steptime_dkt_ks'][self.cur_sub_goal] > 0.9 or \
                    done or (self.sub_episode_running_steps == self.sub_episode_max_length - 1 and observation[1]):
                sub_reward = mindspore.Tensor([1.0]).view(1, -1)
                sub_done = mindspore.Tensor([1.0]).view(1, -1)
                self.sub_episode_running = False
            else:
                sub_reward = mindspore.Tensor([0.0]).view(1, -1)
                sub_done = mindspore.Tensor([0.0]).view(1, -1)
        else:
            raise ValueError('reward low level setting error')

        self.sub_episode[3] = mds_concat((self.sub_episode[3], sub_reward), 0)
        self.sub_episode[4] = mds_concat((self.sub_episode[4], sub_done), 0)

        self.sub_episode[2]['states'] = mds_concat((self.sub_episode[2]['states'], input_data), 0)
        self.sub_episode[2]['states_lengths_ids'] = mds_concat(
            (self.sub_episode[2]['states_lengths_ids'], states_lengths_ids), 0)
        self.sub_episode[2]['targets'].append(targets[0])
        RNN_concate_vector = self.sub_episode[0]['RNN_concate_vector'][-1]
        self.sub_episode[2]['RNN_concate_vector'] = mds_concat(
            (self.sub_episode[2]['RNN_concate_vector'], RNN_concate_vector.unsqueeze(0)), 0)

        self.sub_episode_running_steps += 1
        if (self.sub_episode_running_steps == self.sub_episode_max_length or
                self.args['step_count'] + 1 >= self.args['max_steps']):
            self.sub_episode_running = False

        # sub_episode ending
        if not self.sub_episode_running:
            self.sub_intinsic_rewards = mds_concat(
                (self.sub_intinsic_rewards, mindspore.ops.sum(self.sub_episode[3], 0)), 0)
            self.all_sub_episodes.append(self.sub_episode)
            # HRL big episode setting
            if done:
                big_reward = mindspore.Tensor([1.0]).view(1, -1)
                big_done = mindspore.Tensor([1.0]).view(1, -1)
            else:
                big_done = mindspore.Tensor([0.0]).view(1, -1)
                if (self.model_settings['reward high level'] == 'env-only' or
                        'rs' in self.model_settings['reward high level']):
                    big_reward = mindspore.Tensor([0.0]).view(1, -1)
                else:
                    raise ValueError('reward high level setting error')

            self.big_episode[3] = mds_concat((self.big_episode[3], big_reward), 0)
            self.big_episode[4] = mds_concat((self.big_episode[4], big_done), 0)

    def get_embedding_candidates(self):
        candidates = self.embedding_candidates[self.cur_sub_goal]
        return candidates
