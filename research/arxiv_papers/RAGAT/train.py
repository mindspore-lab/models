import os
import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
from sklearn.metrics import classification_report

import mindspore
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.common.initializer import Normal

# 设置MindSpore运行模式
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 模型参数
max_seq_len = 128
embed_dim = 128
num_channels = 100
kernel_sizes = [3, 4, 5]
rnn_hidden_dim = 128
num_heads = 4
num_classes = 2
batch_size = 32
num_epochs = 3
learning_rate = 0.001
weight_decay = 1e-4
dropout_rate = 0.5

# 构建邻接矩阵
def build_adj_matrix(tokens, max_seq_len, window_size=2):
    adj = np.eye(max_seq_len, dtype=np.float32)
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window_size, len(tokens))):
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj

# 词汇表构建
def build_vocab(texts):
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        tokens = jieba.lcut(text)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

# 预处理每个文本（返回词索引序列和邻接矩阵）
def preprocess_with_adj(text, vocab, max_seq_len):
    tokens = jieba.lcut(text)
    seq = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    if len(seq) < max_seq_len:
        seq = seq + [vocab["<pad>"]] * (max_seq_len - len(seq))
    else:
        seq = seq[:max_seq_len]
    adj = build_adj_matrix(tokens[:max_seq_len], max_seq_len)
    return seq, adj

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

# 完整模型
class TextCNN_GRU_MHA_BiGCN(nn.Cell):
    def __init__(self, vocab_size, embed_dim, num_channels, kernel_sizes, rnn_hidden_dim,
                 num_heads, num_classes, dropout=0.5):
        super(TextCNN_GRU_MHA_BiGCN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0,
                                      embedding_table=Normal(0.1))
        self.conv_layers = nn.CellList([
            nn.Conv1d(embed_dim, num_channels, k, pad_mode='same', weight_init=Normal(0.1))
            for k in kernel_sizes])
        self.relu = nn.ReLU()
        self.concat = ops.Concat(axis=1)
        self.gru = nn.GRU(input_size=num_channels * len(kernel_sizes), hidden_size=rnn_hidden_dim,
                          num_layers=1, batch_first=True)
        self.mha = MultiHeadSelfAttention(rnn_hidden_dim, num_heads)
        self.bigcn = BiGCN(embed_dim, embed_dim, rnn_hidden_dim)
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.fc = nn.Dense(rnn_hidden_dim * 2, num_classes)

    def construct(self, x, adj):
        x_embed = self.embedding(x)
        x_conv_input = x_embed.transpose(0, 2, 1)
        conv_outs = [self.relu(conv(x_conv_input)) for conv in self.conv_layers]
        x_conv_cat = self.concat(conv_outs).transpose(0, 2, 1)
        gru_out, _ = self.gru(x_conv_cat)
        mha_out = self.mha(gru_out)
        mha_pooled = ops.ReduceMean(keep_dims=False)(mha_out, 1)
        gcn_out = self.bigcn(x_embed, adj)
        gcn_pooled = ops.ReduceMean(keep_dims=False)(gcn_out, 1)
        features = ops.Concat(axis=-1)((mha_pooled, gcn_pooled))
        return self.fc(self.dropout(features))

# 自定义损失函数类
class WithLossCell_Custom(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(WithLossCell_Custom, self).__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, x, adj, label):
        logits = self.backbone(x, adj)
        loss = self.loss_fn(logits, label)
        return loss

# 数据加载
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_train.columns = df_train.columns.str.strip()
df_test.columns = df_test.columns.str.strip()
texts_train = df_train['text_a'].tolist()
labels_train = df_train['label'].tolist()
texts_test = df_test['text_a'].tolist()
labels_test = df_test['label'].tolist()

vocab = build_vocab(texts_train)
X_train, A_train, Y_train = [], [], []
X_test, A_test, Y_test = [], [], []
for text, label in zip(texts_train, labels_train):
    seq, adj = preprocess_with_adj(text, vocab, max_seq_len)
    X_train.append(seq)
    A_train.append(adj)
    Y_train.append(label)
for text, label in zip(texts_test, labels_test):
    seq, adj = preprocess_with_adj(text, vocab, max_seq_len)
    X_test.append(seq)
    A_test.append(adj)
    Y_test.append(label)
X_train = np.array(X_train, dtype=np.int32)
A_train = np.array(A_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.int32)
X_test = np.array(X_test, dtype=np.int32)
A_test = np.array(A_test, dtype=np.float32)
Y_test = np.array(Y_test, dtype=np.int32)

train_dataset = ds.NumpySlicesDataset({"data": X_train, "adj": A_train, "label": Y_train}, shuffle=True).batch(batch_size)
test_dataset = ds.NumpySlicesDataset({"data": X_test, "adj": A_test, "label": Y_test}, shuffle=False).batch(batch_size)

# 模型训练
model = TextCNN_GRU_MHA_BiGCN(len(vocab), embed_dim, num_channels, kernel_sizes,
                              rnn_hidden_dim, num_heads, num_classes, dropout=dropout_rate)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate, weight_decay=weight_decay)
net_with_loss = WithLossCell_Custom(model, loss_fn)
train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
train_network.set_train()

# 训练与评估
for epoch in range(num_epochs):
    total_loss = 0
    pbar = tqdm(train_dataset.create_dict_iterator(), total=train_steps, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        data, adj, label = batch['data'], batch['adj'], batch['label']
        loss = train_network(data, adj, label)
        total_loss += loss.asnumpy()
        pbar.set_postfix({"loss": f"{loss.asnumpy():.4f}"})
    avg_loss = total_loss / train_steps
    acc = evaluate(model, test_dataset)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Test Acc={acc*100:.2f}%")

print("训练完成 ✅")
