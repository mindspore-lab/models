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

import numpy as np
import mindspore
from mindspore import nn
from EduSim.utils import get_proj_path, mds_concat
from .AC import ActorCritic
from EduSim.deep_model import MLPNet, GCNNet, RnnEncoder
from mindspore_gl import GraphField
from EduSim.utils import get_feature_matrix
import math
from EduSim.utils import batch_cat_targets


class PKSD(nn.Cell):
    def __init__(self, pksd_para_dict):
        super().__init__()

        input_dim = pksd_para_dict['input_dim']
        output_dim = pksd_para_dict['output_dim']
        hidden_dim1 = pksd_para_dict['hidden_dim1']
        hidden_dim2 = pksd_para_dict['hidden_dim2']
        env = pksd_para_dict['env']
        lr_policy = pksd_para_dict['lr_policy']
        lr_pec = pksd_para_dict['lr_pec']
        lr_imp = pksd_para_dict['lr_imp']
        gamma = pksd_para_dict['gamma']
        c2 = pksd_para_dict['c2']
        args = pksd_para_dict['args']
        RNN_encoder_output_dim = pksd_para_dict['RNN_encoder_output_dim']

        self.name = 'PKSD'
        self.policy_mode = 'on_policy'
        self.pre_learn_node = '-1'
        self.gamma = gamma
        self.lr_policy = lr_policy
        self.lr_pec = lr_pec
        self.lr_imp = lr_imp
        self.num_skills = len(list(env.action_space))
        self.z_size = 2 * self.num_skills
        self.input_dim = input_dim + self.z_size
        self.output_dim = output_dim
        self.env = env
        self.c2 = c2
        self.episode_count = 0
        self.args = args
        self.big_episode_reward_DKT_states = []
        self.embed_dim = pksd_para_dict['embed_dim']
        self.RNN_encoder_input_dim = pksd_para_dict['RNN_encoder_input_dim']
        self.RNN_encoder_output_dim = RNN_encoder_output_dim
        self.phase_1_epi = int(args['PKSD_phase_1_ratio'] * args['max_episode_num'])
        self.env_graph = self.env.knowledge_structure
        self.n_nodes = len(self.env_graph.nodes)
        self.n_edges = len(self.env_graph.edges)
        edges = list(self.env_graph.edges)
        self.src = mindspore.Tensor([edge[0] for edge in edges], dtype=mindspore.int64)
        self.dst = mindspore.Tensor([edge[1] for edge in edges], dtype=mindspore.int64)
        # 计算每个节点的入度和出度
        in_degree = np.zeros(self.n_nodes, dtype=np.int64)
        out_degree = np.zeros(self.n_nodes, dtype=np.int64)
        for edge in edges:
            out_degree[edge[0]] += 1
            in_degree[edge[1]] += 1
        self.in_degree_tensor = mindspore.Tensor(in_degree, dtype=mindspore.int64)
        self.out_degree_tensor = mindspore.Tensor(out_degree, dtype=mindspore.int64)
        self.env_graph_edges = mindspore.Tensor(list(self.env_graph.edges), dtype=mindspore.int64).t()
        self.graph_field = GraphField(self.src, self.dst, self.n_nodes, self.n_edges)

        self.initial_graph_embeddings = mindspore.Tensor(self.env.graph_embeddings)
        self.graph_embedding_dim = self.initial_graph_embeddings.shape[-1] + 1  # 这里加1是最后一维加上knowledge state

        self.__help_init__(output_dim, hidden_dim1, hidden_dim2)

    def __help_init__(self, output_dim, hidden_dim1, hidden_dim2):
        # log encoder
        rnn_para_dict = {
            'input_size': self.RNN_encoder_input_dim,
            'emb_dim': self.embed_dim,
            'hidden_size': hidden_dim1,
            'num_skills': self.RNN_encoder_output_dim,
            'nlayers': 2,
            'dropout': 0.0,
            'out_activation': 'Tanh'
        }
        self.log_encoder = RnnEncoder(rnn_para_dict)

        # perfect encoder
        if self.args['PKSD_PerEncoder'] == 'MLP':
            mlp_para_dict = {
                'input_dim': self.num_skills,
                'hidden_dim1': hidden_dim1,
                'hidden_dim2': hidden_dim2,
                'output_dim': self.z_size
            }
            self.perfect_encoder = MLPNet(mlp_para_dict)
        elif self.args['PKSD_PerEncoder'] == 'GNN':
            self.perfect_encoder = GCNNet(feat_dim=self.graph_embedding_dim,
                                          hidden_dim_size=32,
                                          n_classes=self.z_size)
        else:
            raise ValueError('Wrong perfect encoder setting')

        # imperfect encoder
        if self.args['PKSD_ImPerEncoder'] == 'RNN':
            imp_enc_rnn_para_dict = {
                'input_size': 2 * self.num_skills,
                'emb_dim': self.num_skills,
                'hidden_size': hidden_dim1,
                'num_skills': self.z_size,
                'nlayers': 2,
                'dropout': 0.0,
                'out_activation': 'Tanh'
            }
            self.imperfect_encoder = RnnEncoder(imp_enc_rnn_para_dict)
        elif self.args['PKSD_ImPerEncoder'] == 'RNN_GNN':
            imp_enc_rnn_para_dict = {
                'input_size': 2 * self.num_skills,
                'emb_dim': self.num_skills,
                'hidden_size': hidden_dim1,
                'num_skills': self.num_skills,
                'nlayers': 2,
                'dropout': 0.0,
                'out_activation': 'Tanh'
            }
            self.imperfect_encoder_R = RnnEncoder(imp_enc_rnn_para_dict)
            self.imperfect_encoder_G = GCNNet(feat_dim=self.graph_embedding_dim,
                                              hidden_dim_size=32,
                                              n_classes=self.z_size)
        else:
            raise ValueError('Wrong imperfect encoder setting')

        # agent
        if self.args['PKSD_base_policy'] == 'AC':
            ac_para_dict = {
                'input_dim': self.input_dim,
                'output_dim': output_dim,
                'hidden_dim1': hidden_dim1,
                'hidden_dim2': hidden_dim2,
                'env': self.env,
                'lr_rate': self.lr_policy,
                'gamma': self.gamma,
                'args': self.args,
            }
            self.base_policy = ActorCritic(ac_para_dict)

        # 定义优化器
        if self.args['PKSD_ImPerEncoder'] == 'RNN_GNN':
            # first part
            self.optimizer_1 = nn.Adam(params=self.perfect_encoder.trainable_params() +
                                              self.imperfect_encoder_R.trainable_params() +
                                              self.imperfect_encoder_G.trainable_params(),
                                       learning_rate=0.0001)
            self.grad_fn_combined = mindspore.value_and_grad(self.construct, None,
                                                             weights=self.perfect_encoder.trainable_params() +
                                                                     self.imperfect_encoder_R.trainable_params() +
                                                                     self.imperfect_encoder_G.trainable_params())
            # second part
            self.optimizer_policy = nn.Adam(params=self.base_policy.policy_net.trainable_params() +
                                                   self.log_encoder.trainable_params(),
                                            learning_rate=self.args['learning_rate'])
            self.grad_fn_policy = mindspore.value_and_grad(self.construct, None,
                                                           weights=self.base_policy.policy_net.trainable_params() +
                                                                   self.log_encoder.trainable_params())
            # third part
            self.optimizer_value = nn.Adam(params=self.base_policy.value_net.trainable_params(),
                                           learning_rate=self.args['learning_rate'] * 10)
            self.grad_fn_value = mindspore.value_and_grad(self.construct, None,
                                                          weights=self.base_policy.value_net.trainable_params())
            # last part
            self.imperfect_encoder_optimizer = nn.Adam(self.imperfect_encoder_R.trainable_params() +
                                                       self.imperfect_encoder_G.trainable_params(),
                                                       learning_rate=0.0001)
            self.grad_fn_reg = mindspore.value_and_grad(self.regress_forward_fn, None,
                                                        weights=self.imperfect_encoder_R.trainable_params() +
                                                       self.imperfect_encoder_G.trainable_params(), has_aux=True)
        else:
            # # first part
            self.optimizer_1 = nn.Adam(params=self.perfect_encoder.trainable_params() +
                                              self.imperfect_encoder.trainable_params(),
                                       learning_rate=0.0001)
            self.grad_fn_combined = mindspore.value_and_grad(self.construct, None,
                                                             weights=self.perfect_encoder.trainable_params() +
                                                                     self.imperfect_encoder.trainable_params())
            # second part
            self.optimizer_policy = nn.Adam(params=self.log_encoder.trainable_params() +
                                                   self.base_policy.policy_net.trainable_params(),
                                            learning_rate=self.args['learning_rate'])
            self.grad_fn_policy = mindspore.value_and_grad(self.construct, None,
                                                           weights=self.log_encoder.trainable_params() +
                                                                   self.base_policy.policy_net.trainable_params())
            # third part
            self.optimizer_value = nn.Adam(params=self.base_policy.value_net.trainable_params(),
                                           learning_rate=self.args['learning_rate'] * 10)
            self.grad_fn_value = mindspore.value_and_grad(self.construct, None,
                                                          weights=self.base_policy.value_net.trainable_params())
            # last part
            self.imperfect_encoder_optimizer = nn.Adam(self.imperfect_encoder.trainable_params(), learning_rate=0.0001)
            self.grad_fn_reg = mindspore.value_and_grad(self.regress_forward_fn, None,
                                                        weights=self.imperfect_encoder.trainable_params(),
                                                        has_aux=True)

    def step(self, states_dict, _):
        # log encoding
        encoded_states = self.log_encoder(states_dict['states'])
        sequence_length = states_dict['states_lengths_ids'].shape[0]
        encoded_states = (encoded_states.gather_elements(0, states_dict['states_lengths_ids'].view(1, -1, 1).
                                                         broadcast_to((1, sequence_length, encoded_states.shape[2]))).
                          squeeze(0))
        states_dict['states'] = batch_cat_targets(encoded_states, states_dict['targets'], self.num_skills)

        # pksd encoding
        current_ks = self.get_learner_ks()
        self.args['steptime_knowledge_state'] = mds_concat((self.args['steptime_knowledge_state'],
                                                            current_ks), axis=0)
        if self.args['episode_count'] < self.phase_1_epi:
            # perfect information encoding
            if self.args['PKSD_PerEncoder'] == 'MLP':
                z_t = self.perfect_encoder(current_ks)
                self.args['steptime_state_saver'] = mds_concat((self.args['steptime_state_saver'],
                                                                current_ks), axis=0)
                if self.args['step_count'] > 0:
                    self.args['steptime_next_state_saver'] = mds_concat((self.args['steptime_next_state_saver'],
                                                                         current_ks), axis=0)
            elif self.args['PKSD_PerEncoder'] == 'GNN':
                cat_ks = current_ks.view(-1).unsqueeze(1)
                input_graph_embeds = mds_concat((self.initial_graph_embeddings, cat_ks), axis=1)
                graph_embeddings = self.perfect_encoder(input_graph_embeds,
                                                        self.in_degree_tensor,
                                                        self.out_degree_tensor,
                                                        *self.graph_field.get_graph())
                z_t = mindspore.ops.mean(graph_embeddings, axis=0).view(1, -1)

                self.args['steptime_state_saver'] = mds_concat((self.args['steptime_state_saver'],
                                                                current_ks), axis=0)
                if self.args['step_count'] > 0:
                    self.args['steptime_next_state_saver'] = mds_concat((self.args['steptime_next_state_saver'],
                                                                         current_ks), axis=0)
            # agent
            """这里的states按理说是RNN 原来RNN的输出，但这里dict里面的是是one hot的"""
            RL_input = mds_concat((states_dict['states'], z_t), axis=1)
            RL_input_dict = {
                'rl_states': RL_input,
                'states_lengths_ids': states_dict['states_lengths_ids'],
                'targets': states_dict['targets']
            }

            action = self.base_policy.step(RL_input_dict,
                                           candidates=[i for i in range(len(list(self.env.action_space)))])
        else:
            imperfect_logs = []
            if self.args['step_count'] > 0:
                imperfect_logs = [[int(log[0]), log[1]] for log in self.env._learner._logs[-self.args['step_count']:]]
            imperfect_log_one_hot = get_feature_matrix(imperfect_logs,
                                                       action_dim=self.num_skills,
                                                       embedding_dim=2 * self.num_skills)
            imperfect_log_one_hot = imperfect_log_one_hot.unsqueeze(0)
            if self.args['PKSD_ImPerEncoder'] == 'RNN':
                imperfect_encoding = self.imperfect_encoder(imperfect_log_one_hot)
                z_t_est = imperfect_encoding[max(0, len(imperfect_logs) - 1), :, :]
            elif self.args['PKSD_ImPerEncoder'] == 'RNN_GNN':
                imperfect_encoding = self.imperfect_encoder_R(imperfect_log_one_hot)
                cat_ks_RNN = imperfect_encoding[max(0, len(imperfect_logs) - 1), :, :].view(-1).unsqueeze(1)

                input_graph_embeds = mds_concat((self.initial_graph_embeddings, cat_ks_RNN), axis=1)
                graph_embeddings = self.imperfect_encoder_G(input_graph_embeds,
                                                            self.in_degree_tensor,
                                                            self.out_degree_tensor,
                                                            *self.graph_field.get_graph())
                z_t_est = mindspore.ops.mean(graph_embeddings, axis=0).view(1, -1)
            # agent
            RL_input = mds_concat((states_dict['states'], z_t_est), axis=1)
            RL_input_dict = {
                'rl_states': RL_input,
                'states_lengths_ids': states_dict['states_lengths_ids'],
                'targets': states_dict['targets']
            }
            action = self.base_policy.step(RL_input_dict,
                                           candidates=[i for i in range(len(list(self.env.action_space)))])
            self.args['steptime_state_saver'] = mds_concat((self.args['steptime_state_saver'],
                                                            imperfect_log_one_hot), axis=0)
            if self.args['step_count'] > 0:
                self.args['steptime_next_state_saver'] = mds_concat((self.args['steptime_next_state_saver'],
                                                                     imperfect_log_one_hot), axis=0)

        return action

    def construct(self, pksd_learn_dict):
        rl_states = pksd_learn_dict['states']
        actions = pksd_learn_dict['actions']
        rl_next_states = pksd_learn_dict['next_states']
        rewards = pksd_learn_dict['rewards']
        dones = pksd_learn_dict['dones']

        # log encoding
        rl_states = self.log_encoder(rl_states)
        sequence_length = pksd_learn_dict['states_lengths_ids'].shape[0]
        rl_states = (rl_states.gather_elements(0, pksd_learn_dict['states_lengths_ids'].view(1, -1, 1).
                                               broadcast_to((1, sequence_length, rl_states.shape[2]))).squeeze(0))
        rl_states = batch_cat_targets(rl_states, pksd_learn_dict['targets'], self.num_skills)

        rl_next_states = self.log_encoder(rl_next_states)
        sequence_length = pksd_learn_dict['next_states_lengths_ids'].shape[0]
        rl_next_states = (rl_next_states.gather_elements(0, pksd_learn_dict['next_states_lengths_ids'].view(1, -1, 1).
                                                         broadcast_to((1, sequence_length, rl_next_states.shape[2])))
                          .squeeze(0))
        rl_next_states = batch_cat_targets(rl_next_states, pksd_learn_dict['targets'], self.num_skills)

        # other parts
        if self.args['episode_count'] < self.phase_1_epi:
            if self.args['steptime_next_state_saver'].shape[0] != self.args['steptime_state_saver'].shape[0]:
                self.args['steptime_next_state_saver'] = mds_concat((self.args['steptime_next_state_saver'],
                                                                     self.get_learner_ks()), axis=0)
            if self.args['PKSD_PerEncoder'] == 'MLP':
                rl_states_sup_tmp = self.perfect_encoder(self.args['steptime_state_saver'])
                rl_next_states_sup_tmp = self.perfect_encoder(self.args['steptime_next_state_saver'])
                rl_states_sup = rl_states_sup_tmp
                rl_next_states_sup = rl_next_states_sup_tmp
            elif self.args['PKSD_PerEncoder'] == 'GNN':
                batch_add_feat = self.args['steptime_state_saver']
                rl_states_sup = self.get_batch_gnn_result(batch_add_feat, perfect_mode=True)

                batch_add_next_feat = self.args['steptime_next_state_saver']
                rl_next_states_sup = self.get_batch_gnn_result(batch_add_next_feat, perfect_mode=True)
            # agent
            rl_states = mds_concat((rl_states, rl_states_sup), axis=1)
            rl_next_states = mds_concat((rl_next_states, rl_next_states_sup), axis=1)

            tmp_rl_states_dict = {
                'rl_states': rl_states,
                'states_lengths_ids': pksd_learn_dict['states_lengths_ids'],
                'targets': pksd_learn_dict['targets']
            }
            tmp_rl_next_states_dict = {
                'rl_states': rl_next_states,
                'states_lengths_ids': pksd_learn_dict['next_states_lengths_ids'],
                'targets': pksd_learn_dict['targets']
            }

            ac_learn_dict = {
                'rl_states_dict': tmp_rl_states_dict,
                'actions': actions,
                'rl_next_states_dict': tmp_rl_next_states_dict,
                'rewards': rewards,
                'dones': dones
            }
            critic_loss, actor_loss = self.base_policy.learn(ac_learn_dict)
        else:
            _, RL_states_sup, RL_next_states_sup = self.regress_forward_fn(cal_loss_flag=False)

            # RL
            pksd_states = mds_concat((rl_states, RL_states_sup), axis=1)
            pksd_next_states = mds_concat((rl_next_states, RL_next_states_sup), axis=1)
            tmp_rl_states_dict = {
                'rl_states': pksd_states,
                'states_lengths_ids': pksd_learn_dict['states_lengths_ids'],
                'targets': pksd_learn_dict['targets']
            }
            tmp_rl_next_states_dict = {
                'rl_states': pksd_next_states,
                'states_lengths_ids': pksd_learn_dict['next_states_lengths_ids'],
                'targets': pksd_learn_dict['targets']
            }
            ac_learn_dict = {
                'rl_states_dict': tmp_rl_states_dict,
                'actions': actions,
                'rl_next_states_dict': tmp_rl_next_states_dict,
                'rewards': rewards,
                'dones': dones
            }
            critic_loss, actor_loss = self.base_policy.learn(ac_learn_dict)
        return critic_loss, actor_loss

    def learn(self, pksd_learn_dict):
        (critic_loss, actor_loss), grads = self.grad_fn_combined(pksd_learn_dict)
        self.optimizer_1(grads)
        (_, _), value_grads = self.grad_fn_value(pksd_learn_dict)
        self.optimizer_value(value_grads)
        (_, _), policy_grads = self.grad_fn_policy(pksd_learn_dict)
        self.optimizer_policy(policy_grads)

        # regress loss
        if self.args['episode_count'] >= self.phase_1_epi:
            (regress_loss, RL_states_sup, RL_next_states_sup), imp_grad = self.grad_fn_reg(True)
            self.imperfect_encoder_optimizer(imp_grad)
        else:
            regress_loss = mindspore.Tensor(0.0)
        return critic_loss, actor_loss, regress_loss


    def get_learner_ks(self):
        ks_ = mindspore.Tensor(self.env._learner._state).view(1, -1)
        if self.env.env_name == 'KSS':
            ks_ = mindspore.ops.sigmoid(ks_)
        return ks_

    def get_batch_gnn_result(self, batch_add_feat, perfect_mode=True):
        batch_outs = mindspore.Tensor([])  # [bz, num_skills, embedsize]
        for b in range(batch_add_feat.shape[0]):
            cat_feat = batch_add_feat[b].unsqueeze(1)
            input_graph_embeds = mds_concat((self.initial_graph_embeddings, cat_feat), axis=1)
            if perfect_mode:
                graph_embeddings = self.perfect_encoder(input_graph_embeds,
                                                        self.in_degree_tensor,
                                                        self.out_degree_tensor,
                                                        *self.graph_field.get_graph()).unsqueeze(0)
            else:
                graph_embeddings = self.imperfect_encoder_G(input_graph_embeds,
                                                            self.in_degree_tensor,
                                                            self.out_degree_tensor,
                                                            *self.graph_field.get_graph()).unsqueeze(0)
            z_b = mindspore.ops.mean(graph_embeddings, axis=1).view(1, -1)
            batch_outs = mds_concat((batch_outs, z_b), axis=0) # average pooling - [bz, z_size]
        return batch_outs

    def regress_forward_fn(self, cal_loss_flag=False):
        # imperfect log information
        imperfect_logs = []
        for i in range(len(self.args['learner_initial_logs']), len(self.env._learner._logs)):
            imperfect_logs.append([int(self.env._learner._logs[i][0]), int(self.env._learner._logs[i][1])])
        imperfect_log_one_hot = get_feature_matrix(imperfect_logs,
                                                   action_dim=self.num_skills,
                                                   embedding_dim=2 * self.num_skills)
        imperfect_log_one_hot = imperfect_log_one_hot.unsqueeze(0)
        if self.args['steptime_next_state_saver'].shape[0] != self.args['steptime_state_saver'].shape[0]:
            self.args['steptime_next_state_saver'] = mds_concat(
                (self.args['steptime_next_state_saver'], imperfect_log_one_hot), axis=0)

        # imperfect encoding
        if self.args['PKSD_ImPerEncoder'] == 'RNN':
            RL_states_sup_tmp = self.imperfect_encoder(self.args['steptime_state_saver'])
            RL_next_states_sup_tmp = self.imperfect_encoder(self.args['steptime_next_state_saver'])

            state_indexs = (mindspore.ops.sum(self.args['steptime_state_saver'], dim=(1, 2)).view(1, -1, 1))
            state_indexs = (state_indexs.
                            broadcast_to((1, state_indexs.shape[1], RL_states_sup_tmp.shape[-1])).long() - 1)

            zero = mindspore.ops.zeros(state_indexs.shape, dtype=mindspore.int64)
            state_indexs = mindspore.ops.where(state_indexs < 0, zero, state_indexs)
            RL_states_sup = RL_states_sup_tmp.gather_elements(dim=0, index=state_indexs).squeeze(0)

            next_state_indexs = (mindspore.ops.sum(self.args['steptime_next_state_saver'], dim=(1, 2))
                                 .view(1, -1, 1))
            next_state_indexs = next_state_indexs.broadcast_to((1, next_state_indexs.shape[1],
                                                                RL_next_states_sup_tmp.shape[-1])).long() - 1
            zero = mindspore.ops.zeros(next_state_indexs.shape, dtype=mindspore.int64)
            next_state_indexs = mindspore.ops.where(next_state_indexs < 0, zero, next_state_indexs)
            RL_next_states_sup = RL_next_states_sup_tmp.gather_elements(dim=0, index=next_state_indexs).squeeze(0)

        elif self.args['PKSD_ImPerEncoder'] == 'RNN_GNN':
            RL_states_sup_tmp = self.imperfect_encoder_R(self.args['steptime_state_saver'])
            RL_next_states_sup_tmp = self.imperfect_encoder_R(self.args['steptime_next_state_saver'])

            """
            state part
            """
            state_indexs = (mindspore.ops.sum(self.args['steptime_state_saver'], dim=(1, 2))
                            .view(1, -1, 1))
            state_indexs = state_indexs.broadcast_to((1, state_indexs.shape[1],
                                                      RL_states_sup_tmp.shape[-1])).long() - 1
            zero = mindspore.ops.zeros(state_indexs.shape, dtype=mindspore.int64)
            state_indexs = mindspore.ops.where(state_indexs < 0, zero, state_indexs)
            RL_states_sup = (RL_states_sup_tmp.gather_elements(dim=0, index=state_indexs)
                             .squeeze(0))

            RL_states_sup = self.get_batch_gnn_result(RL_states_sup, perfect_mode=False)

            """
            next state part
            """
            next_state_indexs = (mindspore.ops.sum(self.args['steptime_next_state_saver'], dim=(1, 2))
                                 .view(1, -1, 1))
            next_state_indexs = next_state_indexs.broadcast_to((1, next_state_indexs.shape[1],
                                                                RL_next_states_sup_tmp.shape[-1])).long() - 1
            zero = mindspore.ops.zeros(next_state_indexs.shape, dtype=mindspore.int64)
            next_state_indexs = mindspore.ops.where(next_state_indexs < 0, zero, next_state_indexs)
            RL_next_states_sup = (RL_next_states_sup_tmp.gather_elements(dim=0, index=next_state_indexs)
                                  .squeeze(0))
            RL_next_states_sup = self.get_batch_gnn_result(RL_next_states_sup, perfect_mode=False)

        # regression
        if cal_loss_flag:
            if self.args['PKSD_PerEncoder'] == 'MLP':
                knowledge_state_encoding = mindspore.ops.stop_gradient(self.perfect_encoder(self.args['steptime_knowledge_state']))
            elif self.args['PKSD_PerEncoder'] == 'GNN':
                batch_ks = self.args['steptime_knowledge_state']
                knowledge_state_encoding = mindspore.ops.stop_gradient(self.get_batch_gnn_result(batch_ks, perfect_mode=True))
            """regression loss部分"""
            self.regress_loss_weight = max(min(math.log10(self.args['episode_count'] / 100), 1.0), 0.0001)
            regress_loss = self.regress_loss_weight * mindspore.ops.mse_loss(RL_states_sup, knowledge_state_encoding)
        else:
            regress_loss = mindspore.Tensor(0.0)
        return regress_loss, RL_states_sup, RL_next_states_sup
