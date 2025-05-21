import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.common.initializer import Normal

# 设置MindSpore运行模式
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# Multi-Head Attention 模块
class MultiHeadSelfAttention(nn.Cell):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_linear = nn.Dense(hidden_dim, hidden_dim)
        self.k_linear = nn.Dense(hidden_dim, hidden_dim)
        self.v_linear = nn.Dense(hidden_dim, hidden_dim)
        self.out_linear = nn.Dense(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(axis=-1)
        self.batch_matmul = ops.BatchMatMul()

    def construct(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        Q = self.q_linear(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = self.k_linear(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = self.v_linear(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = self.batch_matmul(Q, K.transpose(0, 1, 3, 2)) * (1.0 / self.head_dim ** 0.5)
        attn_weights = self.softmax(scores)
        context = self.batch_matmul(attn_weights, V).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)
        return self.out_linear(context)

# BiGCN 模块
class BiGCN(nn.Cell):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiGCN, self).__init__()
        self.gcn_forward = nn.Dense(input_dim, hidden_dim)
        self.gcn_backward = nn.Dense(input_dim, hidden_dim)
        self.out_proj = nn.Dense(hidden_dim * 2, output_dim)
        self.relu = nn.ReLU()

    def construct(self, x, adj):
        h_fwd = ops.BatchMatMul()(adj, x)
        h_fwd = self.relu(self.gcn_forward(h_fwd))
        h_bwd = ops.BatchMatMul()(adj.transpose(0, 2, 1), x)
        h_bwd = self.relu(self.gcn_backward(h_bwd))
        return self.out_proj(ops.Concat(axis=-1)((h_fwd, h_bwd)))

# TextCNN 模块
class TextCNN(nn.Cell):
    def __init__(self, embed_dim, num_channels, kernel_sizes):
        super(TextCNN, self).__init__()
        self.conv_layers = nn.CellList([
            nn.Conv1d(embed_dim, num_channels, k, pad_mode='same', weight_init=Normal(0.1))
            for k in kernel_sizes])
        self.relu = nn.ReLU()

    def construct(self, x):
        x_conv_input = x.transpose(0, 2, 1)
        conv_outs = [self.relu(conv(x_conv_input)) for conv in self.conv_layers]
        return ops.Concat(axis=1)(conv_outs)

# 完整模型结构：TextCNN + GRU + MHA + BiGCN
class TextCNN_GRU_MHA_BiGCN(nn.Cell):
    def __init__(self, vocab_size, embed_dim, num_channels, kernel_sizes, rnn_hidden_dim,
                 num_heads, num_classes, dropout=0.5):
        super(TextCNN_GRU_MHA_BiGCN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0,
                                      embedding_table=Normal(0.1))
        self.textcnn = TextCNN(embed_dim, num_channels, kernel_sizes)
        self.gru = nn.GRU(input_size=num_channels * len(kernel_sizes), hidden_size=rnn_hidden_dim,
                          num_layers=1, batch_first=True)
        self.mha = MultiHeadSelfAttention(rnn_hidden_dim, num_heads)
        self.bigcn = BiGCN(embed_dim, embed_dim, rnn_hidden_dim)
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.fc = nn.Dense(rnn_hidden_dim * 2, num_classes)

    def construct(self, x, adj):
        x_embed = self.embedding(x)
        cnn_out = self.textcnn(x_embed)
        gru_out, _ = self.gru(cnn_out)
        mha_out = self.mha(gru_out)
        gcn_out = self.bigcn(x_embed, adj)
        features = ops.Concat(axis=-1)((mha_out, gcn_out))
        return self.fc(self.dropout(features))
