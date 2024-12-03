import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.text as text
import mindspore.nn as nn
from mindspore import Tensor, context
import numpy as np
from mindspore.train.callback import Callback
from sklearn.metrics import f1_score

# 设置MindSpore运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 数据预处理函数
def preprocess_text(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            sentence = lines[i].strip()
            label = int(lines[i + 1].strip())
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

# 加载训练集和测试集
train_file = './train.txt'
test_file = './test.txt'
train_sentences, train_labels = preprocess_text(train_file)
test_sentences, test_labels = preprocess_text(test_file)

# 假设每个句子最长为100个字符，使用ASCII字符集
max_length = 100
vocab = text.Vocab.from_list([chr(i) for i in range(32, 127)])  # ASCII字符集

# 将字符转换为索引
def sentence_to_indices(sentence, vocab, max_length):
    sentence = sentence[:max_length]  # 截断长句
    indices = [vocab[token] if token in vocab else 0 for token in sentence]
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))  # 填充
    return indices

# 将所有句子转换为索引
train_data = [sentence_to_indices(sentence, vocab, max_length) for sentence in train_sentences]
test_data = [sentence_to_indices(sentence, vocab, max_length) for sentence in test_sentences]

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# 创建MindSpore数据集
train_dataset = ds.NumpySlicesDataset({"data": Tensor(np.array(train_data), ms.int32), 
                                       "label": Tensor(train_labels, ms.int32)}, shuffle=True)
test_dataset = ds.NumpySlicesDataset({"data": Tensor(np.array(test_data), ms.int32), 
                                      "label": Tensor(test_labels, ms.int32)}, shuffle=False)

# 定义简单的CNN模型
class SimpleCNN(nn.Cell):
    def __init__(self, vocab_size, embed_size, num_class):
        super(SimpleCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv = nn.Conv2d(1, 128, kernel_size=(3, embed_size), pad_mode="valid")
        self.pool = nn.MaxPool2d(kernel_size=(max_length - 2, 1))
        self.fc = nn.Dense(128, num_class)
    
    def construct(self, x):
        x = self.embedding(x)  # (batch_size, max_length, embed_size)
        x = x.expand_dims(1)  # (batch_size, 1, max_length, embed_size)
        x = self.conv(x)  # (batch_size, 128, max_length - 2, 1)
        x = self.pool(x)  # (batch_size, 128, 1, 1)
        x = x.squeeze()  # (batch_size, 128)
        x = self.fc(x)  # (batch_size, num_class)
        return x

# 自定义回调函数，用于计算F1分数
class F1Callback(Callback):
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
    
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        logits_list = []
        label_list = []
        for data in self.test_dataset.create_dict_iterator():
            logits = self.model.predict(Tensor(data['data'], ms.int32))
            logits_list.append(np.argmax(logits.asnumpy(), axis=1))
            label_list.append(data['label'].asnumpy())
        
        logits_array = np.concatenate(logits_list)
        labels_array = np.concatenate(label_list)
        f1 = f1_score(labels_array, logits_array, average='macro')
        print(f"F1 Score at epoch {cb_params.cur_epoch_num}: {f1:.4f}")

# 模型训练设置
embed_size = 20
num_class = 2  # 0和1两个类别
vocab_size = len(vocab)

net = SimpleCNN(vocab_size, embed_size, num_class)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(net.trainable_params(), learning_rate=0.001)

model = ms.Model(net, loss_fn, optimizer, metrics={'accuracy'})

# 训练集批处理
batch_size = 32
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# 训练模型并计算F1分数
f1_callback = F1Callback(model, test_dataset)

print("Starting training...")
model.train(5, train_dataset, callbacks=[f1_callback])

# 在测试集上进行评估
print("Evaluating on test dataset...")
acc = model.eval(test_dataset)
print(f"Test accuracy: {acc}")
