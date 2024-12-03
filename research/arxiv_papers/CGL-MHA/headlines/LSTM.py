import mindspore
from mindspore import nn, context
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms as C
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.metrics import f1_score

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


# 文本预处理函数
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens


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
        self.vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common())}
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


# 结合 GRU 和 LSTM 网络
class SentimentNet(nn.Cell):
    def __init__(
            self,
            vocab_size,
            embedding_dim=128,
            hidden_size=128,
            num_classes=5,
            num_layers=1,
            bidirectional=False
    ):
        super(SentimentNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

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

        direction = 2 if bidirectional else 1
        # 合并 GRU 和 LSTM 的输出
        self.fc = nn.Dense(hidden_size * direction * 2, num_classes)

    def construct(self, x):
        x = self.embedding(x)

        # 分别通过 GRU 和 LSTM
        gru_out, _ = self.gru(x)
        lstm_out, _ = self.lstm(x)

        gru_out = gru_out[:, -1, :]
        lstm_out = lstm_out[:, -1, :]
        combined = mindspore.ops.concat((gru_out, lstm_out), axis=1)

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
    ds_generator = GeneratorDataset(
        dataset,
        ["data", "label"],
        shuffle=shuffle
    )
    type_cast_op = C.TypeCast(mindspore.int32)
    ds_generator = ds_generator.map(operations=type_cast_op, input_columns="label")
    ds_generator = ds_generator.batch(batch_size)
    return ds_generator


# 加载数据集
train_dataset = SentimentDataset('train.txt')
test_dataset = SentimentDataset('test.txt', vocab=train_dataset.vocab)


# 定义主函数
def main():
    hyperparams = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'embedding_dim': 128,
        'hidden_size': 128,
        'num_layers': 1,
        'bidirectional': False
    }

    train_ds = create_dataset(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    test_ds = create_dataset(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    vocab_size = len(train_dataset.vocab)
    num_classes = len(set(train_dataset.data['label']))

    # 初始化模型、损失函数和优化器
    net = SentimentNet(
        vocab_size,
        embedding_dim=hyperparams['embedding_dim'],
        hidden_size=hyperparams['hidden_size'],
        num_classes=num_classes,
        num_layers=hyperparams['num_layers'],
        bidirectional=hyperparams['bidirectional']
    )
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=hyperparams['learning_rate'])

    metrics = {"Accuracy": nn.Accuracy(), "F1": F1Metric(num_classes)}
    model = mindspore.Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics)

    print("开始训练...")
    model.train(hyperparams['num_epochs'], train_ds, dataset_sink_mode=False)
    print("训练完成。")

    print("开始评估...")
    eval_metrics = model.eval(test_ds, dataset_sink_mode=False)
    print("模型在测试集上的评估结果:", eval_metrics)


# 运行主函数
if __name__ == "__main__":
    main()
