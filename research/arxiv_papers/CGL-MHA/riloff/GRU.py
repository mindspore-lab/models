import mindspore
from mindspore import nn, context, Tensor
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms as C
import mindspore.dataset as ds
import numpy as np
import pandas as pd
import re
from collections import Counter
import mindspore
from mindspore import nn, context, Tensor
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms as C
import mindspore.dataset as ds
import numpy as np
import pandas as pd
import re
from collections import Counter

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
    # 确保行数为偶数
    assert len(lines) % 2 == 0, "数据文件中的行数应为偶数，每两行组成一个样本。"
    # 每次取两行
    for i in range(0, len(lines), 2):
        text = lines[i]
        label = lines[i+1]
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
        x = x[:, -1, :]  # 取最后一个时间步的输出
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
        self.vocab = {word: idx+1 for idx, (word, _) in enumerate(word_counts.most_common())}
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

# 加载数据集
train_dataset = SentimentDataset('train.txt')
test_dataset = SentimentDataset('test.txt', vocab=train_dataset.vocab)

# 其余代码保持不变...
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
num_classes = 5  # 确保类别数正确
learning_rate = 0.001
num_epochs = 10
embedding_dim = 128
hidden_size = 128

# 初始化模型、损失函数和优化器
net = SentimentNet(vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size, num_classes=num_classes)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)

# 定义模型
model = mindspore.Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={"Accuracy": nn.Accuracy()})

# 训练模型
print("开始训练...")
model.train(num_epochs, train_ds, dataset_sink_mode=False)
print("训练完成。")

# 评估模型
print("开始评估...")
acc = model.eval(test_ds, dataset_sink_mode=False)
print("模型在测试集上的准确率:", acc)
