import mindspore.common.dtype as mstype
from mindspore import Tensor
import numpy as np

import mindspore.nn as nn
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore.common.initializer import initializer, XavierUniform

class GATLayer(nn.Cell):
    def __init__(self, in_features, out_features, num_heads, temperature=1, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout
        self.num_heads = num_heads
        self.out_featuers = out_features
        self.W = nn.Dense(in_features, out_features, has_bias=False)
        self.W.weight.set_data(initializer(XavierUniform(), self.W.weight.shape))
        self.a = nn.Dense(2 * out_features, num_heads)
        self.a.weight.set_data(initializer(XavierUniform(), self.a.weight.shape))
        self.leakyRelu = nn.LeakyReLU()

    def construct(self, input, adj, mask=None):
        batch_size = P.Shape()(input)[0]
        degree = P.Shape()(input)[1]
        input = self.W(input)
        input1 = P.Reshape()(P.Tile()(input, (1, 1, degree)), (batch_size, degree * degree, -1))
        input2 = P.Tile()(input, (1, degree, 1))
        final_input = P.Concat(-1)((input1, input2))
        e = P.Reshape()(P.Transpose()(self.leakyRelu(self.a(final_input)), (0, 2, 1)), (batch_size, self.num_heads, degree, degree))

        zero_vec = Tensor(-9e15 * P.OnesLike()(e))
        adj = P.Reshape()(P.Tile()(adj.unsqueeze(1), (1, self.num_heads, 1, 1)), (batch_size, self.num_heads, degree, degree))
        attention = P.Select()(adj > 0, e, zero_vec)
        attention = nn.Softmax(-1)(attention)
        attention = nn.Dropout(self.dropout)(attention)
        h_prime = P.ReduceMean(keep_dims=False)(ops.matmul(attention, input.unsqueeze(1)), 1)

        return h_prime


class TripleGATLayer(nn.Cell):
    def __init__(self, in_feats, out_feats, num_heads):
        super(TripleGATLayer, self).__init__()
        self.conv1 = GATLayer(in_feats, out_feats, num_heads=num_heads)
        self.conv2 = GATLayer(in_feats, out_feats, num_heads=num_heads)
        self.conv3 = GATLayer(in_feats, out_feats, num_heads=num_heads)
        self.concat = P.Concat(axis=-1)
        self.mean = P.ReduceMean(keep_dims=False)

    def construct(self, input, adj, adj_in, adj_out):
        batch_size = adj.shape[0]
        degree = adj.shape[1]
        h1 = self.conv1(input, adj)
        h2 = self.conv2(input, adj_in)
        h3 = self.conv3(input, adj_out)
        h_concat = self.concat((h1, h2, h3)).view(batch_size, degree, 3, -1)
        h_mean = self.mean(h_concat, axis=2)
        return h_mean
    
if __name__ == '__main__':
    input = np.random.rand(100, 5, 128).astype(np.float32)
    adj = np.array([[1, 0, 1, 0, 0],
                    [0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 0, 0, 1]])
    adj = np.tile(adj[np.newaxis, :, :], (100, 1, 1))
    input = Tensor(input)
    adj = Tensor(adj, dtype=mstype.float32)
    
    gat = TripleGATLayer(128, 256, 5)
    h = gat(input, adj, adj, adj)
    print(h.shape)
    print()