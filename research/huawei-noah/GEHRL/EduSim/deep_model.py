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
import mindspore
from mindspore import nn
from EduSim.utils import batch_cat_targets


# class MLPNet(nn.Module):
#     def __init__(self,
#                  input_dim,
#                  hidden_dim1,
#                  hidden_dim2,
#                  output_dim):
#         super(MLPNet, self).__init__()
#         self.name = 'MLP'
#         self.MLP = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim1),
#             nn.Tanh(),
#             nn.Linear(hidden_dim1, hidden_dim2),
#             nn.Tanh(),
#             nn.Linear(hidden_dim2, output_dim)
#         )
#
#     def forward(self, input):
#         out = self.MLP(input)
#         return torch.sigmoid(out)
#
#

class RnnEncoder(nn.Cell):
    def __init__(self, rnn_encoder_para_dict):
        super().__init__()
        self.name = 'normal_rnn_encoder'
        hidden_size = rnn_encoder_para_dict['hidden_size']
        input_size = rnn_encoder_para_dict['input_size']
        emb_dim = rnn_encoder_para_dict['emb_dim']
        num_skills = rnn_encoder_para_dict['num_skills']
        nlayers = rnn_encoder_para_dict['nlayers']
        dropout = rnn_encoder_para_dict['dropout']
        out_activation = rnn_encoder_para_dict['out_activation']

        self.nhid = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout

        self.embedding_layer = nn.Dense(input_size, emb_dim)
        self.rnn = nn.GRU(input_size, hidden_size, nlayers)
        self.fc_out = nn.Dense(hidden_size, num_skills)
        self.dropout = nn.Dropout(p=self.dropout)
        self.out_activation = out_activation

    def construct(self, x):
        x = x.permute(1, 0, 2)
        h_0, _ = self.init_hidden_state(x.shape[1])

        output, _ = self.rnn(x, h_0)
        out = self.fc_out(output)  # [sequence_length,batch_size,num_skills]
        if self.out_activation == 'Tanh':
            out = mindspore.ops.tanh(out)
        elif self.out_activation == 'Sigmoid':
            out = mindspore.ops.sigmoid(out)
        return out

    def init_hidden_state(self, batch_size):
        h_0 = mindspore.ops.rand((self.nlayers, batch_size, self.nhid))
        c_0 = mindspore.ops.rand((self.nlayers, batch_size, self.nhid))
        return h_0, c_0


#
#
# class Qnet(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, num_skills, dueling_dqn, noisy_dqn):
#         super(Qnet, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.fc_out = nn.Linear(hidden_size2, num_skills)
#
#         self.fc_A = nn.Linear(hidden_size2, num_skills)
#         self.fc_V = nn.Linear(hidden_size2, 1)
#
#         if noisy_dqn:
#             self.fc1 = NoisyLinear(input_size, hidden_size1)
#             self.fc2 = NoisyLinear(hidden_size1, hidden_size2)
#
#             self.fc_out = NoisyLinear(hidden_size2, num_skills)
#
#             self.fc_A = NoisyLinear(hidden_size2, num_skills)
#             self.fc_V = NoisyLinear(hidden_size2, 1)
#
#         self.dueling_dqn = dueling_dqn
#         self.noisy_dqn = noisy_dqn
#
#     def forward(self, x):
#         if self.dueling_dqn:
#             # A = self.fc_A(out)
#             # V = self.fc_V(out)
#             A = self.fc_A(F.relu(self.fc2(F.relu(self.fc1(x)))))
#             V = self.fc_V(F.relu(self.fc2(F.relu(self.fc1(x)))))
#
#             Q = V + A - A.mean(1).view(-1, 1)
#             return Q
#
#         else:
#             x = torch.relu(self.fc1(x))
#             x = torch.relu(self.fc2(x))
#             out = self.fc_out(x)
#             return out
#
#
class PolicyNet(nn.Cell):
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, action_dim):
        super().__init__()
        self.actor_net = nn.SequentialCell(
            nn.Dense(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dense(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dense(hidden_dim2, action_dim),
        )

        self.fc1 = nn.Dense(state_dim, hidden_dim1)
        self.fc2 = nn.Dense(hidden_dim1, hidden_dim2)
        self.fc_out = nn.Dense(hidden_dim2, action_dim)

    def construct(self, state):
        dist = self.actor_net(state)
        dist = mindspore.ops.softmax(dist, axis=1)
        return dist


class ValueNet(nn.Cell):
    def __init__(self, state_dim, hidden_dim1, hidden_dim2):
        super().__init__()
        self.value_net = nn.SequentialCell(
            nn.Dense(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dense(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dense(hidden_dim2, 1),
        )

        self.fc1 = nn.Dense(state_dim, hidden_dim1)
        self.fc2 = nn.Dense(hidden_dim1, hidden_dim2)
        self.fc_out = nn.Dense(hidden_dim2, 1)

    def construct(self, state):
        value = self.value_net(state)
        return value


class PolicyNetWithOutterEncoder(nn.Cell):
    def __init__(self, para_dict):
        super().__init__()
        state_dim = para_dict['state_dim']
        hidden_dim1 = para_dict['hidden_dim1']
        hidden_dim2 = para_dict['hidden_dim2']
        action_dim = para_dict['action_dim']
        outer_encoder = para_dict['outer_encoder']
        RLInputCompo = para_dict['RLInputCompo']

        self.outer_encoder = outer_encoder
        self.actor_net = nn.SequentialCell(
            nn.Dense(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dense(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dense(hidden_dim2, action_dim),
        )
        self.action_dim = action_dim
        self.RLInputCompo = RLInputCompo

    def construct(self, state_input_dict):
        states = state_input_dict['states']
        states_lengths_ids = state_input_dict['states_lengths_ids']
        targets = state_input_dict['targets']

        encoded_states = self.outer_encoder(states)

        # rnn version
        sequence_length = states_lengths_ids.shape[0]
        encoded_states = (encoded_states.gather_elements(0, states_lengths_ids.view(1, -1, 1).
                                                         broadcast_to((1, sequence_length, encoded_states.shape[2]))).
                          squeeze(0))
        if self.RLInputCompo == 'RNNwithTarget':
            RL_states = batch_cat_targets(encoded_states, targets, self.action_dim)
        elif self.RLInputCompo == 'RNNSubgoalTarget':
            RNN_concate_vector = state_input_dict['RNN_concate_vector']
            RL_states = mindspore.ops.cat((encoded_states, RNN_concate_vector), -1)
            RL_states = batch_cat_targets(RL_states, targets, self.action_dim)
        else:
            raise ValueError('wrong setting of RLInputCompo')

        # mlp version
        # RL_states = encoded_states

        dist = self.actor_net(RL_states)
        dist = mindspore.ops.softmax(dist, axis=1)
        return dist


class ValueNetWithOutterEncoder(nn.Cell):
    def __init__(self, para_dict):
        super().__init__()
        state_dim = para_dict['state_dim']
        hidden_dim1 = para_dict['hidden_dim1']
        hidden_dim2 = para_dict['hidden_dim2']
        action_dim = para_dict['action_dim']
        outer_encoder = para_dict['outer_encoder']
        RLInputCompo = para_dict['RLInputCompo']

        self.value_net = nn.SequentialCell(
            nn.Dense(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dense(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dense(hidden_dim2, 1),
        )
        self.outer_encoder = outer_encoder
        self.RLInputCompo = RLInputCompo
        self.action_dim = action_dim

    def construct(self, state_input_dict):
        states = state_input_dict['states']
        states_lengths_ids = state_input_dict['states_lengths_ids']
        targets = state_input_dict['targets']

        encoded_states = self.outer_encoder(states)

        # rnn version
        sequence_length = states_lengths_ids.shape[0]
        encoded_states = encoded_states.gather_elements(0, states_lengths_ids.view(1, -1, 1).broadcast_to(
            (1, sequence_length, encoded_states.shape[2]))).squeeze(0)
        if self.RLInputCompo == 'RNNwithTarget':
            RL_states = batch_cat_targets(encoded_states, targets, self.action_dim)
        elif self.RLInputCompo == 'RNNSubgoalTarget':
            RNN_concate_vector = state_input_dict['RNN_concate_vector']
            RL_states = mindspore.ops.cat((encoded_states, RNN_concate_vector), -1)
            RL_states = batch_cat_targets(RL_states, targets, self.action_dim)
        else:
            raise ValueError('wrong setting of RLInputCompo')

        # mlp version
        # RL_states = encoded_states

        value = self.value_net(RL_states)
        return value


class DKTnet(nn.Cell):
    def __init__(self, dkt_para_dict):
        super().__init__()
        input_size = dkt_para_dict['input_size']
        emb_dim = dkt_para_dict['emb_dim']
        hidden_size = dkt_para_dict['hidden_size']
        num_skills = dkt_para_dict['num_skills']
        nlayers = dkt_para_dict['nlayers']
        dropout = dkt_para_dict['dropout']

        self.name = 'DKT'
        self.nhid = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout

        self.embedding_layer = nn.Dense(input_size, emb_dim, weight_init="normal", bias_init="zeros")

        self.rnn = nn.LSTM(emb_dim, hidden_size, nlayers)
        self.fc_out = nn.Dense(hidden_size, num_skills)

        self.dropout = nn.Dropout(p=self.dropout)

    def construct(self, x):
        x = x.permute(1, 0, 2)
        h_0, c_0 = self.init_hidden_state(x.shape[1])

        embed = self.embedding_layer(x)
        output, _ = self.rnn(embed, (h_0, c_0))
        out = self.fc_out(output)  # [sequence_length,batch_size,num_skills]
        out = self.dropout(out)
        return out

    def init_hidden_state(self, batch_size):
        h_0 = mindspore.ops.rand((self.nlayers, batch_size, self.nhid))
        c_0 = mindspore.ops.rand((self.nlayers, batch_size, self.nhid))
        return h_0, c_0

#
# '''
# 构建模型，使用两层GCN，第一层GCN使得节点特征矩阵
#     (N, in_channel) -> (N, out_channel)
# 第二层GCN直接输出
#     (N, out_channel) -> (N, num_class)
# 激活函数使用relu函数，网络最后对节点的各个类别score使用softmax归一化。
# '''
#
#
# class GCNNet(nn.Module):
#     def __init__(self, feat_dim, num_class, num_node=None):
#         super(GCNNet, self).__init__()
#         self.conv1 = GCNConv(feat_dim, 128)
#         self.conv2 = GCNConv(128, num_class)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#
#         return x
#
#
# class GATNet(torch.nn.Module):
#     def __init__(self, in_channel, out_channel, node_num=None):
#         super(GATNet, self).__init__()
#         self.gat1 = GATConv(in_channel, 8, 8, dropout=0.6)
#         self.gat2 = GATConv(64, out_channel, 1, dropout=0.6)
#
#     def forward(self, x, edge_index):
#         x = self.gat1(x, edge_index)
#         x = self.gat2(x, edge_index)
#         return x
