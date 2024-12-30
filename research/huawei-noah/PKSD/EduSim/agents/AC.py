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
import mindspore
from mindspore import nn
from EduSim.deep_model import PolicyNet, ValueNet


class ActorCritic(nn.Cell):
    def __init__(self, ac_para_dict):
        super(ActorCritic, self).__init__()
        input_dim = ac_para_dict['input_dim']
        output_dim = ac_para_dict['output_dim']
        hidden_dim1 = ac_para_dict['hidden_dim1']
        hidden_dim2 = ac_para_dict['hidden_dim2']
        env = ac_para_dict['env']
        lr_rate = ac_para_dict['lr_rate']
        gamma = ac_para_dict['gamma']
        args = ac_para_dict['args']

        self.name = 'ac'
        self.policy_mode = 'on_policy'
        self.pre_learn_node = '0'  # int(0) for kes
        self.gamma = gamma
        self.learning_rate = lr_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.env = env
        self.args = args

        para_dict = {
            'state_dim': self.input_dim,
            'hidden_dim1': hidden_dim1,
            'hidden_dim2': hidden_dim2,
            'action_dim': self.output_dim,
        }

        self.policy_net = PolicyNet(para_dict)
        self.value_net = ValueNet(para_dict)

        # 优化器设置
        # self.grad_fn_policy = mindspore.value_and_grad(self.forward_fn_policy,
        #                                                None, weights=self.policy_net.trainable_params())
        # self.grad_fn_value = mindspore.value_and_grad(self.forward_fn_value, None,
        #                                               weights=self.value_net.trainable_params())
        # self.policy_optimizer = mindspore.nn.Adam(self.policy_net.trainable_params(), learning_rate=self.learning_rate)
        # self.value_opitimizer = mindspore.nn.Adam(self.value_net.trainable_params(),
        #                                           learning_rate=self.learning_rate * 8)

    def step(self, state_input_dict, candidates):
        probs = self.policy_net(state_input_dict)

        candidate_probs = probs.gather_elements(dim=1, index=mindspore.Tensor(candidates).view(1, -1))
        # candidate_probs = mindspore.ops.softmax(candidate_probs)
        # candidate_dist = mindspore.nn.probability.distribution.Categorical(probs=candidate_probs)
        # id = candidate_dist.sample()
        # action = candidates[id]

        action = random.choices(candidates, weights=candidate_probs.view(-1).asnumpy(), k=1)[0]

        return action

    def forward_fn_policy(self, advantages, RL_states_dict, actions):
        probs = self.policy_net(RL_states_dict)
        # [batch_size, action_sim] -> [batch_size, 1]
        log_probs = probs.gather_elements(dim=1, index=actions).view(-1, 1).log()
        # log_probs = mindspore.ops.log(probs.gather_elements(dim=1, index=actions)).view(-1, 1)
        actor_loss = mindspore.ops.mean(-log_probs * advantages)
        return actor_loss

    def forward_fn_value(self, td_target, RL_states_dict):
        output = self.value_net(RL_states_dict)
        critic_loss = mindspore.ops.mse_loss(output, td_target)
        return critic_loss

    def learn(self, ac_learn_dict):
        rl_states_dict = ac_learn_dict['rl_states_dict']
        actions = ac_learn_dict['actions']
        rl_next_states_dict = ac_learn_dict['rl_next_states_dict']
        rewards = ac_learn_dict['rewards']
        dones = ac_learn_dict['dones']

        td_target = rewards + self.gamma * mindspore.ops.stop_gradient(self.value_net(rl_next_states_dict) * (1 - dones))    # 时序差分目标
        G = 0
        for i in reversed(range(rewards.shape[0])):
            G = self.gamma * G + rewards[i]
            td_target[i] = G
        advantages = td_target - mindspore.ops.stop_gradient(self.value_net(rl_states_dict))  # 时序差分误差

        critic_loss = self.forward_fn_value(td_target, rl_states_dict)
        policy_loss = self.forward_fn_policy(advantages, rl_states_dict, actions)
        return critic_loss, policy_loss
