import mindspore
from mindspore import nn, context
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms as c
import numpy as np
import pandas as pd
import re
from collections import Counter

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


# 文本预处理
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens


# 数据加载
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


class SentimentNet(nn.Cell):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, num_classes=5):
        super(SentimentNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Dense(hidden_size, num_classes)

    def construct(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


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


# 创建数据集对象
def create_dataset(dataset, batch_size=32, shuffle=True):
    ds_generator = GeneratorDataset(
        dataset,
        ["data", "label"],
        shuffle=shuffle
    )
    type_cast_op = c.TypeCast(mindspore.int32)
    ds_generator = ds_generator.map(operations=type_cast_op, input_columns="label")
    ds_generator = ds_generator.batch(batch_size)
    return ds_generator


# 模型训练
def train_and_evaluate_model(train_dataset, test_dataset):
    # 超参数
    params = {
        "num_classes": 5,
        "learning_rate": 0.001,
        "num_epochs": 10,
        "embedding_dim": 128,
        "hidden_size": 128,
        "batch_size": 32
    }

    # 创建数据集对象
    train_ds = create_dataset(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True
    )
    test_ds = create_dataset(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False
    )

    # 获取词汇表大小
    vocab_size = len(train_dataset.vocab)

    # 初始化模型、损失函数和优化器
    net = SentimentNet(
        vocab_size,
        embedding_dim=params["embedding_dim"],
        hidden_size=params["hidden_size"],
        num_classes=params["num_classes"]
    )
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=params["learning_rate"])

    # 定义模型
    model = mindspore.Model(
        net,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics={"Accuracy": nn.Accuracy()}
    )

    # 训练模型
    print("开始训练...")
    model.train(
        params["num_epochs"],
        train_ds,
        dataset_sink_mode=False
    )
    print("训练完成。")

    # 评估模型
    print("开始评估...")
    acc = model.eval(
        test_ds,
        dataset_sink_mode=False
    )
    print("模型在测试集上的准确率:", acc)

    return model, acc


# 导入数据集
train_dataset = SentimentDataset('train.txt')
test_dataset = SentimentDataset('test.txt', vocab=train_dataset.vocab)


def main():
    model, acc = train_and_evaluate_model(train_dataset, test_dataset)
    print("模型训练和评估流程已完成。")


if __name__ == "__main__":
    main()
