import mindspore
from mindspore import nn, context
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms as C
import mindspore.ops as ops
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.metrics import f1_score
import os
import time
from mindspore.train.callback import Callback  # 从正确的模块导入 Callback

# 设置运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 文本预处理函数
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens

# 新的数据加载函数
def read_text_label_file(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    assert len(lines) % 2 == 0, "数据文件中的行数应为偶数，每两行组成一个样本。"
    for i in range(0, len(lines), 2):
        text = lines[i]
        label = lines[i + 1]
        texts.append(text)
        labels.append(int(label))
    data = pd.DataFrame({'text': texts, 'label': labels})
    return data

# 数据集生成器
class SentimentDataset:
    def __init__(self, data_file, vocab=None, max_len=128):
        self.data = read_text_label_file(data_file)
        self.max_len = max_len
        self.vocab = vocab
        if self.vocab is None:
            self.build_vocab()

    def build_vocab(self):
        all_tokens = []
        for text in self.data['text']:
            tokens = preprocess_text(text)
            all_tokens.extend(tokens)
        word_counts = Counter(all_tokens)
        self.vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.items())}
        self.vocab['<PAD>'] = 0

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']
        label = int(label)
        tokens = preprocess_text(text)
        token_ids = [self.vocab.get(token, 0) for token in tokens]
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        else:
            token_ids += [0] * (self.max_len - len(token_ids))
        token_ids = np.array(token_ids, dtype=np.int32)
        return token_ids, label

    def __len__(self):
        return len(self.data)

# 加载预训练的词向量
def load_pretrained_embeddings(embedding_path, vocab, embedding_dim):
    embeddings_index = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split()
            word = ' '.join(values[:-embedding_dim])
            coefs = np.asarray(values[-embedding_dim:], dtype='float32')
            embeddings_index[word] = coefs
    print(f'共加载了 {len(embeddings_index)} 个词向量。')

    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix

# 定义多头注意力机制
class MultiHeadAttention(nn.Cell):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0, "嵌入维度必须能被头数整除。"
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Dense(embed_dim, embed_dim)
        self.k_linear = nn.Dense(embed_dim, embed_dim)
        self.v_linear = nn.Dense(embed_dim, embed_dim)
        self.out_proj = nn.Dense(embed_dim, embed_dim)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.softmax = ops.Softmax(axis=-1)
        self.sqrt_head_dim = mindspore.Tensor(np.sqrt(self.head_dim), mindspore.float32)

    def construct(self, query, key, value):
        batch_size, seq_len, embed_dim = query.shape
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        Q = self.reshape(Q, (batch_size, seq_len, self.num_heads, self.head_dim))
        Q = self.transpose(Q, (0, 2, 1, 3))
        K = self.reshape(K, (batch_size, seq_len, self.num_heads, self.head_dim))
        K = self.transpose(K, (0, 2, 1, 3))
        V = self.reshape(V, (batch_size, seq_len, self.num_heads, self.head_dim))
        V = self.transpose(V, (0, 2, 1, 3))
        scores = ops.BatchMatMul()(Q, self.transpose(K, (0, 1, 3, 2))) / self.sqrt_head_dim
        attn_weights = self.softmax(scores)
        attn_output = ops.BatchMatMul()(attn_weights, V)
        attn_output = self.transpose(attn_output, (0, 2, 1, 3))
        attn_output = self.reshape(attn_output, (batch_size, seq_len, embed_dim))
        output = self.out_proj(attn_output)
        return output, attn_weights

# 定义模型，结合 CNN、GRU 和 LSTM 网络，并加入多头注意力机制
class SentimentNet(nn.Cell):
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=128, num_classes=5,
                 num_layers=1, bidirectional=False, embedding_matrix=None,
                 num_filters=128, filter_sizes=(3, 4, 5), max_len=128, num_heads=8):
        super(SentimentNet, self).__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            embedding_tensor = mindspore.Tensor(embedding_matrix, mindspore.float32)
            self.embedding.embedding_table.set_data(embedding_tensor)

        self.convs = nn.CellList([
            nn.SequentialCell([
                nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs, pad_mode='pad'),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.compute_pool_size(fs))
            ]) for fs in filter_sizes
        ])

        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                          has_bias=True, batch_first=True, bidirectional=bidirectional)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                            has_bias=True, batch_first=True, bidirectional=bidirectional)
        
        self.bidirectional = bidirectional
        direction = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.direction = direction
        self.multihead_attn = MultiHeadAttention(embed_dim=hidden_size * direction * 2, num_heads=num_heads)

        total_feature_size = num_filters * len(filter_sizes) + hidden_size * direction * 2
        self.fc = nn.Dense(total_feature_size, num_classes)

    def compute_pool_size(self, kernel_size):
        return self.max_len - kernel_size + 1

    def construct(self, x):
        x = self.embedding(x)
        embedded = x
        x_cnn = embedded.transpose(0, 2, 1)
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x_cnn)
            conv_out = conv_out.squeeze(2)
            conv_outputs.append(conv_out)
        cnn_out = ops.concat(conv_outputs, 1)

        gru_out, _ = self.gru(embedded)
        lstm_out, _ = self.lstm(embedded)
        combined_out = ops.concat((gru_out, lstm_out), 2)
        attn_out, attn_weights = self.multihead_attn(combined_out, combined_out, combined_out)
        attn_out_pooled = ops.reduce_mean(attn_out, 1)
        combined_features = ops.concat((cnn_out, attn_out_pooled), 1)
        out = self.fc(combined_features)
        return out

# 自定义 F1 值计算类
class F1Metric(nn.Metric):
    def __init__(self, num_classes):
        super(F1Metric, self).__init__()
        self.num_classes = num_classes
        self.clear()

    def clear(self):
        self._y_true = []
        self._y_pred = []

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[0])
        y_true = self._convert_data(inputs[1])
        if isinstance(y_pred, mindspore.Tensor):
            y_pred = y_pred.asnumpy()
        if isinstance(y_true, mindspore.Tensor):
            y_true = y_true.asnumpy()
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        self._y_true.extend(y_true)
        self._y_pred.extend(y_pred)

    def eval(self):
        f1 = f1_score(self._y_true, self._y_pred, average='macro')
        return f1

from mindspore.train.callback import Callback

# 自定义回调函数保存指标
from mindspore.train.callback import Callback
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 自定义回调函数保存指标
class SaveMetricsCallback(Callback):
    def __init__(self, model_name, file_path="metrics_log.txt"):
        super(SaveMetricsCallback, self).__init__()
        self.model_name = model_name
        self.file_path = file_path
        self.start_time = time.time()

    def on_train_epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        print(f"Epoch {cb_params.cur_epoch_num} begins.")

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        # 直接获取标量损失值
        loss = cb_params.net_outputs.asnumpy()  # 获取损失值
        cur_step = cb_params.cur_step_num

        # 仅记录损失值和批次编号
        with open(self.file_path, "a") as f:
            f.write(f"Step: {cur_step}, Loss: {loss:.4f}\n")

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        time_taken = time.time() - self.start_time
        print(f"Epoch {epoch} ends. Time taken: {time_taken:.2f}s.")


# 加载数据集
train_dataset = SentimentDataset('train.txt')
test_dataset = SentimentDataset('test.txt', vocab=train_dataset.vocab, max_len=train_dataset.max_len)

embedding_dim = 100
embedding_path = 'glove.6B.100d.txt'
if not os.path.exists(embedding_path):
    print(f"预训练词向量文件 {embedding_path} 不存在，请下载并放置在指定路径。")
    exit(1)

embedding_matrix = load_pretrained_embeddings(embedding_path, train_dataset.vocab, embedding_dim)

def create_dataset(dataset, batch_size=32, shuffle=True):
    ds_generator = GeneratorDataset(dataset, ["data", "label"], shuffle=shuffle)
    type_cast_op = C.TypeCast(mindspore.int32)
    ds_generator = ds_generator.map(operations=type_cast_op, input_columns="label")
    ds_generator = ds_generator.batch(batch_size)
    return ds_generator

train_ds = create_dataset(train_dataset, batch_size=32, shuffle=True)
test_ds = create_dataset(test_dataset, batch_size=32, shuffle=False)

# 定义超参数
vocab_size = len(train_dataset.vocab)
num_classes = len(set(train_dataset.data['label']))
learning_rate = 0.001
num_epochs = 20
hidden_size = 128
num_layers = 1
bidirectional = False
num_filters = 128
filter_sizes = (3, 4, 5)
max_len = train_dataset.max_len
num_heads = 8

net = SentimentNet(vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size,
                   num_classes=num_classes, num_layers=num_layers, bidirectional=bidirectional,
                   embedding_matrix=embedding_matrix, num_filters=num_filters, filter_sizes=filter_sizes,
                   max_len=max_len, num_heads=num_heads)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)

metrics = {"Accuracy": nn.Accuracy(), "F1": F1Metric(num_classes)}
model = mindspore.Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics)

model_name = "SentimentNet_Model"
file_path = f"{model_name}_metrics_log.txt"
save_metrics_cb = SaveMetricsCallback(model_name, file_path)

print("开始训练...")
model.train(num_epochs, train_ds, callbacks=[save_metrics_cb], dataset_sink_mode=False)
print("训练完成。")

print("开始评估...")
eval_metrics = model.eval(test_ds, dataset_sink_mode=False)
print("模型在测试集上的评估结果:", eval_metrics)
