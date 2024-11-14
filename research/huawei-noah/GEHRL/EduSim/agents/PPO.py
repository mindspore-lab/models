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
import random
import numpy as np
import mindspore
from mindspore import nn
from EduSim.deep_model import PolicyNetWithOutterEncoder, ValueNetWithOutterEncoder
from EduSim.utils import mean_entropy_cal


class PPO(nn.Cell):
    def __init__(self, ppo_para_dict):
        super(PPO, self).__init__()
        input_dim = ppo_para_dict['input_dim']
        output_dim = ppo_para_dict['output_dim']
        hidden_dim1 = ppo_para_dict['hidden_dim1']
        hidden_dim2 = ppo_para_dict['hidden_dim2']
        env = ppo_para_dict['env']
        policy_clip = ppo_para_dict['policy_clip']
        gae_lambda = ppo_para_dict['gae_lambda']
        k_epochs = ppo_para_dict['k_epochs']
        mini_batch_size = ppo_para_dict['mini_batch_size']
        lr_rate = ppo_para_dict['lr_rate']
        gamma = ppo_para_dict['gamma']
        c2 = ppo_para_dict['c2']
        args = ppo_para_dict['args']
        outer_encoder = ppo_para_dict['outer_encoder']
        RLInputCompo = ppo_para_dict['RLInputCompo']

        self.name = 'ppo'
        self.policy_mode = 'on_policy'
        self.pre_learn_node = '-1'
        self.gamma = gamma
        self.learning_rate = lr_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_dim = output_dim
        self.env = env
        self.policy_clip = policy_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.mini_batch_size = mini_batch_size
        self.c2 = c2
        self.args = args

        # 替代buffer保存old probs
        self.ep_old_logprobs = []
        self.continuous_normal_sample = mindspore.Tensor([])

        # net设置
        if outer_encoder is not None:
            para_dict = {
                'state_dim': self.input_dim,
                'hidden_dim1': hidden_dim1,
                'hidden_dim2': hidden_dim2,
                'action_dim': self.output_dim,
                'outer_encoder': outer_encoder,
                'RLInputCompo': RLInputCompo
            }
            self.policy_net = PolicyNetWithOutterEncoder(para_dict)
            self.value_net = ValueNetWithOutterEncoder(para_dict)
        else:
            raise ValueError('outer encoder missed!')
        self.MSEloss = nn.MSELoss()

        # 优化器设置
        self.grad_fn_policy = mindspore.value_and_grad(self.forward_fn_policy,
                                                       None, weights=self.policy_net.trainable_params())
        self.grad_fn_value = mindspore.value_and_grad(self.forward_fn_value,
                                                      None, weights=self.value_net.trainable_params())
        self.policy_optimizer = mindspore.nn.Adam(self.policy_net.trainable_params(), learning_rate=self.learning_rate)
        self.value_opitimizer = mindspore.nn.Adam(self.value_net.trainable_params(),
                                                  learning_rate=self.learning_rate * 8)

    def step(self, state_input_dict, candidates):
        probs = self.policy_net(state_input_dict)
        candidate_probs = probs.gather_elements(dim=1, index=mindspore.Tensor(candidates).view(1, -1))

        # candidate_dist = mindspore.nn.probability.distribution.Categorical(probs=candidate_probs)
        # id = candidate_dist.sample()
        # action = candidates[id]

        action = random.choices(candidates, weights=candidate_probs.view(-1).asnumpy(), k=1)[0]
        # 保存old_log_probs
        action_logprob = mindspore.ops.log(probs[0][action])
        self.ep_old_logprobs.append(action_logprob.asnumpy().item())

        return action

    def forward_fn_policy(self, ffn_input_dict):
        batch_states_dict = ffn_input_dict['batch_states_dict']
        batch_actions = ffn_input_dict['batch_actions']
        batch_old_logprobs = ffn_input_dict['batch_old_logprobs']
        advantages = ffn_input_dict['advantages']
        batch_indices = ffn_input_dict['batch_indices']

        probs = self.policy_net(batch_states_dict)
        batch_new_logprobs = mindspore.ops.log(probs.gather_elements(dim=1, index=batch_actions)).squeeze(1)
        # batch_dist = mindspore.nn.probability.distribution.Categorical(probs=probs)
        # batch_new_logprobs = batch_dist.log_prob(mindspore.ops.squeeze(batch_actions))

        ratios = mindspore.ops.exp(batch_new_logprobs - batch_old_logprobs)
        batch_advantages = mindspore.ops.squeeze(advantages[batch_indices])
        surr1 = ratios * batch_advantages
        surr2 = mindspore.ops.clamp(ratios, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages

        # 为了数值稳定
        a = mindspore.ops.cat((surr1.view(1, -1), surr2.view(1, -1)), axis=0)
        a, _ = mindspore.ops.min(a, axis=0)
        actor_loss = -a.mean()

        entropy_loss = -self.c2 * mean_entropy_cal(probs)  # 计算不慢，但是梯度计算很慢
        # batch_dist = mindspore.nn.probability.distribution.Categorical(probs=probs)
        # entropy_loss = -self.c2 * batch_dist.entropy().mean()
        return actor_loss, entropy_loss

    def forward_fn_value(self, batch_states_dict, returns, batch_indices):
        batch_state_values = self.value_net(batch_states_dict)
        value_loss = 0.5 * self.MSEloss(batch_state_values, returns[batch_indices])
        return value_loss

    def learn(self, ac_learn_dict, multiepi_sample_ids: list = None, multiepi_done=False, forOutterRNN=False):
        RL_states_dict = ac_learn_dict['RL_states_dict']
        actions = ac_learn_dict['actions']
        rewards = ac_learn_dict['rewards']
        dones = ac_learn_dict['dones']

        # 多个episode连着训练：
        if multiepi_sample_ids is not None:
            tmp_epi_old_logprobs = self.ep_old_logprobs[multiepi_sample_ids[0]:multiepi_sample_ids[1]]
        else:
            tmp_epi_old_logprobs = self.ep_old_logprobs

        # 为整个traj的每个时刻t计算advantages
        values = self.value_net(RL_states_dict)
        ad_input_dict = {
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'gamma': self.gamma,
            'lam': self.gae_lambda
        }
        advantages, returns = self.estimate_advantages(ad_input_dict=ad_input_dict)

        # 分割mini_batches
        ep_length = len(rewards)
        batch_starts = np.arange(0, ep_length, self.mini_batch_size)
        indices = np.arange(ep_length, dtype=np.int64)
        mini_batch_indices = [list(indices[i:i + self.mini_batch_size]) for i in batch_starts]

        # 以mini_batches的形式训练k_epochs
        loss = mindspore.Tensor([0.0])
        for _ in range(self.k_epochs):
            for batch_indices in mini_batch_indices:  # train for k epochs
                batch_indices = mindspore.Tensor(batch_indices)
                # batch_states = RL_states[batch_indices]
                batch_states_dict = {
                    'states': RL_states_dict['states'][batch_indices],
                    'states_lengths_ids': RL_states_dict['states_lengths_ids'][batch_indices],
                    'targets': [RL_states_dict['targets'][i] for i in batch_indices]
                }
                if 'RNN_concate_vector' in RL_states_dict.keys():
                    batch_states_dict['RNN_concate_vector'] = RL_states_dict['RNN_concate_vector'][batch_indices]
                batch_old_logprobs = mindspore.Tensor(tmp_epi_old_logprobs)[batch_indices]
                batch_actions = actions[batch_indices]

                _, grad_value = self.grad_fn_value(batch_states_dict, returns, batch_indices)
                ffn_para_dict = {
                    'batch_states_dict': batch_states_dict,
                    'batch_actions': batch_actions,
                    'batch_old_logprobs': batch_old_logprobs,
                    'advantages': advantages,
                    'batch_indices': batch_indices
                }
                (_, _), grad_policy = self.grad_fn_policy(ffn_para_dict)
                self.value_opitimizer(grad_value)
                self.policy_optimizer(grad_policy)

        if ((multiepi_sample_ids is not None and multiepi_done) or multiepi_sample_ids is None) and forOutterRNN:
            self.ep_old_logprobs = []
        return loss

    def estimate_advantages(self, ad_input_dict):
        rewards = ad_input_dict['rewards']
        dones = ad_input_dict['dones']
        values = ad_input_dict['values']
        gamma = ad_input_dict['gamma']
        lam = ad_input_dict['lam']

        deltas = mindspore.ops.zeros((rewards.shape[0], 1))
        advantages = mindspore.ops.zeros((rewards.shape[0], 1))

        prev_value = 0.0
        prev_advantage = 0.0
        for i in reversed(range(rewards.shape[0])):
            deltas[i] = rewards[i] + gamma * prev_value * (1.0 - dones[i]) - values[i]
            advantages[i] = deltas[i] + gamma * lam * prev_advantage * (1.0 - dones[i])

            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]

        returns = values + advantages

        return advantages, returns
