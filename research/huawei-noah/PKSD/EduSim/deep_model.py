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

from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell
from mindspore_gl.nn import GCNConv


class MLPNet(nn.Cell):
    def __init__(self, para_dict):
        super().__init__()
        input_dim = para_dict['input_dim']
        hidden_dim1 = para_dict['hidden_dim1']
        hidden_dim2 = para_dict['hidden_dim2']
        output_dim = para_dict['output_dim']

        self.mlp_net = nn.SequentialCell(
            nn.Dense(input_dim, hidden_dim1),
            nn.Tanh(),
            nn.Dense(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Dense(hidden_dim2, output_dim),
        )

    def construct(self, input):
        out = self.mlp_net(input)
        return mindspore.ops.sigmoid(out)

class GCNNet(GNNCell):
    """ GCN Net """
    def __init__(self,
                 feat_dim: int,
                 hidden_dim_size: int,
                 n_classes: int):
        super().__init__()
        self.layer0 = GCNConv(feat_dim, hidden_dim_size)
        self.layer1 = GCNConv(hidden_dim_size, n_classes)

    def construct(self, x, in_deg, out_deg, g: Graph):
        """GCN Net forward"""
        x = self.layer0(x, in_deg, out_deg, g)
        x = self.layer1(x, in_deg, out_deg, g)
        return x

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


class PolicyNet(nn.Cell):
    def __init__(self, para_dict):
        super().__init__()
        state_dim = para_dict['state_dim']
        hidden_dim1 = para_dict['hidden_dim1']
        hidden_dim2 = para_dict['hidden_dim2']
        action_dim = para_dict['action_dim']

        self.actor_net = nn.SequentialCell(
            nn.Dense(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dense(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dense(hidden_dim2, action_dim),
        )
        self.action_dim = action_dim

    def construct(self, state_input_dict):
        rl_states = state_input_dict['rl_states']
        dist = self.actor_net(rl_states)
        dist = mindspore.ops.softmax(dist, axis=1)
        return dist


class ValueNet(nn.Cell):
    def __init__(self, para_dict):
        super().__init__()
        state_dim = para_dict['state_dim']
        hidden_dim1 = para_dict['hidden_dim1']
        hidden_dim2 = para_dict['hidden_dim2']
        action_dim = para_dict['action_dim']

        self.value_net = nn.SequentialCell(
            nn.Dense(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dense(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dense(hidden_dim2, 1),
        )
        self.action_dim = action_dim

    def construct(self, state_input_dict):
        rl_states = state_input_dict['rl_states']
        value = self.value_net(rl_states)
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
