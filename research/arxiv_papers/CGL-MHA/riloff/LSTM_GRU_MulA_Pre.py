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

# 设置运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 文本预处理函数
def preprocess_text(text):
    # 转小写
    text = text.lower()
    # 去除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 按空格分词
    tokens = text.split()
    return tokens

# 新的数据加载函数
def read_text_label_file(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 去除空行和首尾空白字符
    lines = [line.strip() for line in lines if line.strip()]
    # 确保行数为偶数，每两行组成一个样本
    assert len(lines) % 2 == 0, "数据文件中的行数应为偶数，每两行组成一个样本。"
    # 每次取两行，第一行是文本，第二行是对应的标签
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
        # 构建词汇表，保留所有词
        self.vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.items())}
        self.vocab['<PAD>'] = 0  # 添加填充符

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']
        label = int(label)  # 标签已经是整数，无需调整

        # 文本预处理
        tokens = preprocess_text(text)

        # 将单词映射为索引
        token_ids = [self.vocab.get(token, 0) for token in tokens]

        # 截断或填充序列
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
    print('共加载了 %s 个词向量。' % len(embeddings_index))

    # 初始化嵌入矩阵
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # 对于未找到的词，可以使用随机初始化或保持为零向量
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix

# 定义模型，结合 GRU 和 LSTM 网络，并加入注意力机制
class SentimentNet(nn.Cell):
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=128, num_classes=5,
                 num_layers=1, bidirectional=False, num_heads=4, embedding_matrix=None):
        super(SentimentNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            # 将预训练的嵌入矩阵转换为 MindSpore Tensor
            embedding_tensor = mindspore.Tensor(embedding_matrix, mindspore.float32)
            # 将嵌入矩阵赋值给嵌入层的参数
            self.embedding.embedding_table.set_data(embedding_tensor)

        # 定义 GRU 和 LSTM
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                          has_bias=True, batch_first=True, bidirectional=bidirectional)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                            has_bias=True, batch_first=True, bidirectional=bidirectional)
        
        self.bidirectional = bidirectional
        direction = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.direction = direction
        self.num_heads = num_heads

        # 定义注意力机制
        self.attention = nn.SequentialCell([
            nn.Dense(hidden_size * direction * 2, hidden_size * direction),
            nn.Tanh(),
            nn.Dense(hidden_size * direction, 1)
        ])

        # 全连接层
        self.fc = nn.Dense(hidden_size * direction * 2, num_classes)  # 乘以2是因为 GRU 和 LSTM 的输出拼接

    def construct(self, x):
        x = self.embedding(x)
        
        # 分别通过 GRU 和 LSTM
        gru_out, _ = self.gru(x)  # gru_out: (batch_size, seq_len, hidden_size * direction)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size * direction)
        
        # 拼接 GRU 和 LSTM 的输出
        combined_out = ops.concat((gru_out, lstm_out), 2)  # 使用位置参数 axis=2
        
        # 注意力机制
        attn_weights = self.attention(combined_out)  # (batch_size, seq_len, 1)
        attn_weights = ops.softmax(attn_weights, 1)  # 对时间步维度做 softmax

        # 加权求和
        attn_applied = combined_out * attn_weights  # (batch_size, seq_len, hidden_size * direction * 2)
        attn_out = ops.reduce_sum(attn_applied, 1)  # (batch_size, hidden_size * direction * 2)

        # 全连接层
        out = self.fc(attn_out)
        return out

# 自定义 F1 值计算类
class F1Metric(nn.Metric):
    def __init__(self, num_classes):
        super(F1Metric, self).__init__()
        self.num_classes = num_classes
        self.clear()

    def clear(self):
        """清除内部状态"""
        self._y_true = []
        self._y_pred = []

    def update(self, *inputs):
        """更新内部状态"""
        y_pred = self._convert_data(inputs[0])
        y_true = self._convert_data(inputs[1])

        # 如果 y_pred 是 MindSpore Tensor，则转换为 NumPy 数组
        if isinstance(y_pred, mindspore.Tensor):
            y_pred = y_pred.asnumpy()
        if isinstance(y_true, mindspore.Tensor):
            y_true = y_true.asnumpy()

        # 如果 y_pred 是 logits，需要转换为预测类别
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        self._y_true.extend(y_true)
        self._y_pred.extend(y_pred)

    def eval(self):
        """计算最终的 F1 值"""
        f1 = f1_score(self._y_true, self._y_pred, average='macro')
        return f1

# 加载数据集
train_dataset = SentimentDataset('train.txt')
test_dataset = SentimentDataset('test.txt', vocab=train_dataset.vocab)

# 加载预训练的词向量
embedding_dim = 100  # GloVe 词向量的维度
embedding_path = 'glove.6B.100d.txt'  # 请确保路径正确
if not os.path.exists(embedding_path):
    print(f"预训练词向量文件 {embedding_path} 不存在，请下载并放置在指定路径。")
    exit(1)

embedding_matrix = load_pretrained_embeddings(embedding_path, train_dataset.vocab, embedding_dim)

# 创建数据集对象
def create_dataset(dataset, batch_size=32, shuffle=True):
    ds_generator = GeneratorDataset(dataset, ["data", "label"], shuffle=shuffle)
    # 类型转换
    type_cast_op = C.TypeCast(mindspore.int32)
    ds_generator = ds_generator.map(operations=type_cast_op, input_columns="label")
    ds_generator = ds_generator.batch(batch_size)
    return ds_generator

train_ds = create_dataset(train_dataset, batch_size=32, shuffle=True)
test_ds = create_dataset(test_dataset, batch_size=32, shuffle=False)

# 定义超参数
vocab_size = len(train_dataset.vocab)
num_classes = len(set(train_dataset.data['label']))  # 确保类别数正确
learning_rate = 0.001
num_epochs = 10
hidden_size = 128
num_layers = 1
bidirectional = False

# 初始化模型、损失函数和优化器
net = SentimentNet(vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size,
                   num_classes=num_classes, num_layers=num_layers, bidirectional=bidirectional,
                   embedding_matrix=embedding_matrix)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)

# 定义模型，添加 F1Metric
metrics = {"Accuracy": nn.Accuracy(), "F1": F1Metric(num_classes)}
model = mindspore.Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics)

# 训练模型
print("开始训练...")
model.train(num_epochs, train_ds, dataset_sink_mode=False)
print("训练完成。")

# 评估模型
print("开始评估...")
eval_metrics = model.eval(test_ds, dataset_sink_mode=False)
print("模型在测试集上的评估结果:", eval_metrics)
