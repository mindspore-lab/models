import mindspore.nn as nn

from mindspore.nn.probability.distribution import Categorical
import mindspore.ops.operations as P
import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor
from utils import *
from src.gat import TripleGATLayer
from src.self_attn import *


class TripleGAT(nn.Cell):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TripleGAT, self).__init__()
        self.conv1 = TripleGATLayer(in_feats, hid_feats, 5)
        self.conv2 = TripleGATLayer(hid_feats, out_feats, 5)
        # self.bn1 = nn.BatchNorm1d(hid_feats)
        # self.bn2 = nn.BatchNorm1d(out_feats)
        self.relu = nn.ReLU()

    def construct(self, adj, adj_in, adj_out, inputs):
        batch = adj.shape[0]
        degree = adj.shape[-1]

        h = self.conv1(inputs, adj, adj_in, adj_out)
        # h = self.bn1(h.view(batch*degree, -1)).view(batch, degree, -1)
        h = self.relu(h)
        h = self.conv2(h, adj, adj_in, adj_out)
        # h = self.bn2(h.view(batch*degree, -1)).view(batch, degree, -1)
        return h


class Actor(nn.Cell):
    def __init__(self):
        super(Actor, self).__init__()
        # embedder args
        self.d_input = 3
        self.d_model = 128
        self.embedder = Embedder(self.d_input, self.d_model)

        # encoder args
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        # feedforward layer inner
        self.d_inner = 512
        self.d_unit = 256
        self.encoder = Encoder(self.num_stacks, self.num_heads,
                               self.d_k, self.d_v, self.d_model, self.d_inner)

        # decoder args
        self.ptr = Pointer(self.d_model, self.d_model)
        # TODO: 需要考虑用于初始化的GAT是否需要与用于更新的GAT共享一个权重？
        self.gat = TripleGAT(self.d_model, self.d_model, self.d_model)
        self.fc = nn.SequentialCell([
            nn.Dense(2 * self.d_model, self.d_model),
            nn.BatchNorm1d(self.d_model)
        ])
        self.g_emb = Glimpse(self.d_model, self.d_unit)
        self.softmax = nn.Softmax()

    def construct(self, inputs: Tensor, deterministic: bool = False):
        """
        :param inputs: numpy.ndarray [batch_size * degree * 2]
        :param deterministic:
        :return:
        """
        batch_size = inputs.shape[0]
        degree = inputs.shape[1]

        adj = P.Fill()(P.DType()(inputs), (batch_size, degree, degree), 1)
        adj_in = P.Fill()(P.DType()(inputs), (batch_size, degree, degree), 1)
        adj_out = P.Fill()(P.DType()(inputs), (batch_size, degree, degree), 1)

        visited = P.ZerosLike()(adj[:,0,:])
        visited[:, 0] = 1

        indexes, log_probs = [], []
        #为inputs添加第三维标记
        pad = P.Fill()(P.DType()(inputs), (batch_size, degree, 1), 0)
        pad[:, 0, 0] = 1
        inputs = P.Concat(-1)((inputs, pad))
        embedings = self.embedder(inputs)
        encodings = self.encoder(embedings, None)
        graph_mask = None
        step = None
        for step in range(degree-1):
            visited_rep = P.Reshape()(P.Tile()(visited, (1, degree)), (batch_size, degree, degree)).astype(ms.int32)
            mask_check = (visited_rep | P.Transpose()(visited_rep, (0, 2, 1))) - (visited_rep & P.Transpose()(visited_rep, (0, 2, 1)))
            if ops.all((mask_check.int() == 0).int()):
                break
            node_embedding = self.gat(adj, adj_in, adj_out, encodings)
            input1 = P.Reshape()(P.Tile()(node_embedding, (1, 1, degree)), (batch_size, degree * degree, -1))
            input2 = P.Tile()(node_embedding, (1, degree, 1))
            final_input = P.Concat(-1)((input1, input2)).reshape(batch_size * degree * degree, -1)
            edge_embedding = self.fc(final_input).reshape(batch_size, degree * degree, -1)
            node_embedding = node_embedding.reshape(batch_size, degree, -1)
            graph_embedding = self.g_emb(node_embedding, graph_mask)
            logits, t = self.ptr(edge_embedding, graph_embedding, mask_check)
            probs = self.softmax(logits)
            distr = Categorical(probs)
            if deterministic:
                _, edge_idx = ops.max(logits, -1)
            else:
                edge_idx = distr.sample()
            x_idx = ops.stop_gradient(ops.div(edge_idx, degree))
            y_idx = ops.stop_gradient(ops.fmod(edge_idx, degree))
            
            indexes.append(x_idx)
            indexes.append(y_idx)
            log_p = distr.log_prob(edge_idx)
            # TODO : detach
            log_p[t] = log_p[t].detach()
            log_probs.append(log_p)
            # 更新visited
            visited = P.ScatterNdUpdate()(visited, P.Stack()([x_idx, y_idx], 1), 1)
            adj = P.TensorScatterUpdate()(adj, P.Stack()([P.Range(batch_size), x_idx, y_idx]), Tensor([1]))

            tmp = adj[P.Range(batch_size), x_idx]
            tmp[P.Range(batch_size), x_idx] = 0
            tmp = P.Reshape()(P.Tile()(tmp.unsqueeze(1), (1, degree, 1)))
            tmp = tmp * P.Transpose()(tmp, (0, 2, 1))
            adj_in = adj_in | tmp

            tmp = adj.transpose(2, 1)[P.Range(batch_size), y_idx]
            tmp[P.Range(batch_size), y_idx] = 0
            tmp = P.Reshape()(P.Tile()(tmp.unsqueeze(1), (1, degree, 1)))
            tmp = tmp * P.Transpose()(tmp, (0, 2, 1))
            adj_out = adj_out | tmp

        log_probs = sum(log_probs)
        return adj, log_probs, indexes
    

class Critic(nn.Cell):

    def __init__(self):
        super(Critic, self).__init__()

        # embedder args
        self.d_input = 3
        self.d_model = 128

        # encoder args
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        self.d_inner = 512
        self.d_unit = 256

        self.crit_embedder = Embedder(self.d_input, self.d_model)
        self.crit_encoder = Encoder(self.num_stacks, self.num_heads, self.d_k, self.d_v, self.d_model, self.d_inner)
        # self.gnn = TripleGNN(self.d_model, self.d_model, self.d_model)
        self.glimpse = Glimpse(self.d_model, self.d_unit)
        self.crit_l1 = nn.Dense(self.d_model, self.d_unit)
        self.crit_l2 = nn.Dense(self.d_unit, 1)
        self.relu = P.ReLU()

    def construct(self, inputs):
        pad = P.Fill()(P.DType()(inputs), (inputs.shape[0], inputs.shape[1], 1), 0)

        pad[:, 0, 0] = 1
        inputs = P.Concat(-1)((inputs, pad))

        critic_encode = self.crit_encoder(self.crit_embedder(inputs), None)
        # critic_encode = self.gnn(adj, adj_in, adj_out, critic_encode)
        glimpse = self.glimpse(critic_encode)
        critic_inner = self.relu(self.critic_l1(glimpse))
        predictions = self.relu(self.critic_l2(critic_inner)).squeeze(-1)

        return predictions
