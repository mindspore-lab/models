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

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


# 文本预处理函数
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens


# 数据加载函数
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
    print('共加载了 %s 个词向量。' % len(embeddings_index))
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix


# 结合 CNN、GRU 和 LSTM 网络，并加入注意力机制
class SentimentNet(nn.Cell):
    def __init__(
            self,
            vocab_size,
            embedding_dim=100,
            hidden_size=128,
            num_classes=5,
            num_layers=1,
            bidirectional=False,
            embedding_matrix=None,
            num_filters=128,
            filter_sizes=(3, 4, 5),
            max_len=128
    ):
        super(SentimentNet, self).__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            embedding_tensor = mindspore.Tensor(embedding_matrix, mindspore.float32)
            self.embedding.embedding_table.set_data(embedding_tensor)

        # 定义 CNN 模块
        self.convs = nn.CellList([
            nn.SequentialCell([
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                    pad_mode='pad'
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.compute_pool_size(fs))
            ]) for fs in filter_sizes
        ])

        # 定义 GRU 和 LSTM
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            has_bias=True,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            has_bias=True,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.bidirectional = bidirectional
        direction = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.direction = direction

        # 定义注意力机制
        self.attention = nn.SequentialCell([
            nn.Dense(hidden_size * direction * 2, hidden_size * direction),
            nn.Tanh(),
            nn.Dense(hidden_size * direction, 1)
        ])

        # 计算全连接层的输入维度
        total_feature_size = num_filters * len(filter_sizes) + hidden_size * direction * 2
        self.fc = nn.Dense(total_feature_size, num_classes)

    def compute_pool_size(self, kernel_size):

        return self.max_len - kernel_size + 1

    def construct(self, x):
        x = self.embedding(x)
        embedded = x

        # CNN 模块
        x_cnn = embedded.transpose(0, 2, 1)
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x_cnn)
            conv_out = conv_out.squeeze(2)
            conv_outputs.append(conv_out)
        cnn_out = ops.concat(conv_outputs, 1)

        # RNN 模块
        gru_out, _ = self.gru(embedded)
        lstm_out, _ = self.lstm(embedded)

        # 拼接 GRU 和 LSTM 的输出
        combined_out = ops.concat((gru_out, lstm_out), 2)

        # 注意力机制
        attn_weights = self.attention(combined_out)
        attn_weights = ops.softmax(attn_weights, 1)  # 对时间步维度做 softmax

        # 加权求和
        attn_applied = combined_out * attn_weights
        attn_out = ops.reduce_sum(attn_applied, 1)

        # 融合 CNN 和 RNN 的特征
        combined_features = ops.concat((cnn_out, attn_out), 1)

        # 全连接层
        out = self.fc(combined_features)
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

        if isinstance(y_pred, mindspore.Tensor):
            y_pred = y_pred.asnumpy()
        if isinstance(y_true, mindspore.Tensor):
            y_true = y_true.asnumpy()
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        self._y_true.extend(y_true)
        self._y_pred.extend(y_pred)

    def eval(self):
        """计算最终的 F1 值"""
        f1 = f1_score(self._y_true, self._y_pred, average='macro')
        return f1


# 创建数据集对象
def create_dataset(dataset, batch_size=32, shuffle=True):
    ds_generator = GeneratorDataset(dataset, ["data", "label"], shuffle=shuffle)
    # 类型转换
    type_cast_op = C.TypeCast(mindspore.int32)
    ds_generator = ds_generator.map(operations=type_cast_op, input_columns="label")
    ds_generator = ds_generator.batch(batch_size)
    return ds_generator


def run_training_and_evaluation(params):
    # 加载数据集
    train_dataset = SentimentDataset(params['train_file'])
    test_dataset = SentimentDataset(
        params['test_file'],
        vocab=train_dataset.vocab,
        max_len=train_dataset.max_len
    )

    # 检查预训练词向量文件
    if not os.path.exists(params['embedding_path']):
        print(f"预训练词向量文件 {params['embedding_path']} 不存在，请下载并放置在指定路径。")
        return

    # 加载预训练的词向量
    embedding_matrix = load_pretrained_embeddings(params['embedding_path'], train_dataset.vocab,
                                                  params['embedding_dim'])

    # 创建数据集对象
    train_ds = create_dataset(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_ds = create_dataset(test_dataset, batch_size=params['batch_size'], shuffle=False)

    # 定义超参数
    vocab_size = len(train_dataset.vocab)
    num_classes = len(set(train_dataset.data['label']))
    max_len = train_dataset.max_len

    # 初始化模型、损失函数和优化器
    net = SentimentNet(
        vocab_size,
        embedding_dim=params['embedding_dim'],
        hidden_size=params['hidden_size'],
        num_classes=num_classes,
        num_layers=params['num_layers'],
        bidirectional=params['bidirectional'],
        embedding_matrix=embedding_matrix,
        num_filters=params['num_filters'],
        filter_sizes=params['filter_sizes'],
        max_len=max_len
    )
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=params['learning_rate'])

    # 定义模型，添加 F1Metric
    metrics = {"Accuracy": nn.Accuracy(), "F1": F1Metric(num_classes)}
    model = mindspore.Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics)

    # 训练模型
    print("开始训练...")
    model.train(params['num_epochs'], train_ds, dataset_sink_mode=False)
    print("训练完成。")

    # 评估模型
    print("开始评估...")
    eval_metrics = model.eval(test_ds, dataset_sink_mode=False)
    print("模型在测试集上的评估结果:", eval_metrics)


if __name__ == "__main__":
    params = {
        'train_file': 'train.txt',
        'test_file': 'test.txt',
        'embedding_path': 'glove.6B.100d.txt',
        'embedding_dim': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'hidden_size': 128,
        'num_layers': 1,
        'bidirectional': False,
        'num_filters': 128,
        'filter_sizes': (3, 4, 5)
    }
    run_training_and_evaluation(params)
