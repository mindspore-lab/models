import mindspore
from mindspore import nn, context
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms as C
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.metrics import f1_score  # 确保已安装 scikit-learn

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
        self.vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common())}
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

# 定义模型，结合 GRU 和 LSTM 网络
class SentimentNet(nn.Cell):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, num_classes=5, num_layers=1, bidirectional=False):
        super(SentimentNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 定义 GRU 和 LSTM
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                          has_bias=True, batch_first=True, bidirectional=bidirectional)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                            has_bias=True, batch_first=True, bidirectional=bidirectional)
        
        direction = 2 if bidirectional else 1
        # 合并 GRU 和 LSTM 的输出，所以全连接层输入尺寸需要调整
        self.fc = nn.Dense(hidden_size * direction * 2, num_classes)  # 乘以2是因为 GRU 和 LSTM 的输出拼接

    def construct(self, x):
        x = self.embedding(x)
        
        # 分别通过 GRU 和 LSTM
        gru_out, _ = self.gru(x)
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        gru_out = gru_out[:, -1, :]  # shape: (batch_size, hidden_size * direction)
        lstm_out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size * direction)
        
        # 拼接 GRU 和 LSTM 的输出
        combined = mindspore.ops.concat((gru_out, lstm_out), axis=1)  # shape: (batch_size, hidden_size * direction * 2)
        
        # 全连接层
        out = self.fc(combined)
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
embedding_dim = 128
hidden_size = 128
num_layers = 1
bidirectional = False

# 初始化模型、损失函数和优化器
net = SentimentNet(vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size,
                   num_classes=num_classes, num_layers=num_layers, bidirectional=bidirectional)
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
