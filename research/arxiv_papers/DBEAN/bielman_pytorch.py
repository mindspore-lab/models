import re
import numpy as np
import random
from collections import Counter
from nltk.corpus import wordnet, stopwords
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# nltk.download('stopwords')
# nltk.download('wordnet')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

try:
    from datasets import load_dataset

    # 加载 AG_NEWS 数据集
    dataset = load_dataset("ag_news")

    # 提取训练和测试数据
    train_data = list(dataset["train"])  # [(label, text), ...]
    test_data = list(dataset["test"])

    print(f"Loaded AG_NEWS dataset successfully! Train size: {len(train_data)}, Test size: {len(test_data)}")
except Exception as e:
    raise RuntimeError("Failed to load AG_NEWS dataset. Ensure 'datasets' is installed. Error: " + str(e))


# 分离出文本和标签列表
# 提取训练数据的文本和标签
train_texts = []
train_labels = []
print("Processing training data:")
for item in tqdm(train_data, desc="Train Data Processing", unit="sample"):
    train_texts.append(item["text"])
    train_labels.append(int(item["label"]))

# 提取测试数据的文本和标签
test_texts = []
test_labels = []
print("Processing test data:")
for item in tqdm(test_data, desc="Test Data Processing", unit="sample"):
    test_texts.append(item["text"])
    test_labels.append(int(item["label"]))  

val_size = 10000
if len(train_texts) > val_size:
    val_texts = train_texts[:val_size]
    val_labels = train_labels[:val_size]
    train_texts = train_texts[val_size:]
    train_labels = train_labels[val_size:]
else:
    val_texts, val_labels = [], []

stop_words = set(stopwords.words('english'))  # 获取英语停用词表
def preprocess_text(text):
    # 转小写
    text = text.lower()
    # 移除标点和数字
    text = re.sub(r'[^a-z\s]', ' ', text)
    # 将多个空格合一
    text = re.sub(r'\s+', ' ', text).strip()
    # 分词
    words = text.split()
    # 去除停用词
    words = [w for w in words if w not in stop_words]
    return words

train_tokens = [preprocess_text(txt) for txt in train_texts]
val_tokens = [preprocess_text(txt) for txt in val_texts] if val_texts else []
test_tokens = [preprocess_text(txt) for txt in test_texts]

def synonym_replace(words, n=1):
    """随机选择n个非停用词，用同义词替换"""
    new_words = words.copy()
    available_indices = [i for i, w in enumerate(new_words) if w not in stop_words]
    random.shuffle(available_indices)
    num_replaced = 0
    for idx in available_indices:
        word = new_words[idx]
        syns = wordnet.synsets(word)
        # 找到与原词词性相同的同义词，如果有的话
        if syns:
            # 获取第一个同义词的词根（lemma），避免特殊字符
            syn_words = [lemma.name() for lemma in syns[0].lemmas() if lemma.name().lower() != word]
            if syn_words:
                new_words[idx] = syn_words[0].replace('_', ' ')  # 用下划线连接的短语替换空格
                num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

def random_deletion(words, p=0.2):
    """以概率p随机删除非停用词"""
    if len(words) == 0:
        return words
    new_words = []
    for w in words:
        if w in stop_words:
            new_words.append(w)  # 停用词直接保留，不删除太多信息
        else:
            if random.uniform(0, 1) > p:  # 以(1-p)的概率保留词汇
                new_words.append(w)
    # 如果全删光了，随机保留一个词
    if len(new_words) == 0:
        new_words.append(random.choice(words))
    return new_words

# 对训练tokens应用数据增强（每条可能产生1条增强句子）
aug_train_tokens = []
aug_train_labels = []
for tokens, label in zip(train_tokens, train_labels):
    # 保留原始
    aug_train_tokens.append(tokens)
    aug_train_labels.append(label)
    # 随机选择增强方式
    if len(tokens) > 0:
        aug_type = random.choice(["sr", "rd"])  # 同义词替换 or 随机删除
        if aug_type == "sr":
            new_tokens = synonym_replace(tokens, n=1)
        else:
            new_tokens = random_deletion(tokens, p=0.1)
        aug_train_tokens.append(new_tokens)
        aug_train_labels.append(label)

# 更新训练集为原始+增强的数据
train_tokens = aug_train_tokens
train_labels = aug_train_labels

# 统计训练集词频
word_counter = Counter()
for tokens in train_tokens:
    word_counter.update(tokens)

# 定义特殊标记
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
# 初始化词->索引映射，保证 <pad>=0, <unk>=1
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for word, freq in word_counter.items():
    # 这里可设定min_freq阈值，过滤低频词; 为简便我们不设过滤
    if word not in vocab:
        vocab[word] = len(vocab)  # 给新词分配新的索引

vocab_size = len(vocab)
pad_idx = vocab[PAD_TOKEN]  # 0
unk_idx = vocab[UNK_TOKEN]  # 1

# 定义函数：将词序列转为索引序列
def tokens_to_ids(tokens):
    return [vocab.get(w, unk_idx) for w in tokens]

# 将所有数据集转换为索引序列，并进行填充/截断
MAX_LEN = 100  # 设定最大序列长度
def pad_or_truncate(seq, max_len=MAX_LEN):
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        return seq + [pad_idx] * (max_len - len(seq))

train_ids = [pad_or_truncate(tokens_to_ids(seq)) for seq in train_tokens]
val_ids = [pad_or_truncate(tokens_to_ids(seq)) for seq in val_tokens] if val_tokens else []
test_ids = [pad_or_truncate(tokens_to_ids(seq)) for seq in test_tokens]

# 将列表转换为numpy数组，便于构建Dataset
X_train = np.array(train_ids, dtype=np.int64)
y_train = np.array(train_labels, dtype=np.int64)
X_val = np.array(val_ids, dtype=np.int64) if val_tokens else None
y_val = np.array(val_labels, dtype=np.int64) if val_tokens else None
X_test = np.array(test_ids, dtype=np.int64)
y_test = np.array(test_labels, dtype=np.int64)

embed_dim = 100
# 下载并解析GloVe文件（glove.6B.100d.txt）
# 假设已经将 glove.6B.100d.txt 放在当前目录
glove_path = "glove.6B.100d.txt"
embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embed_dim))  # 未命中词随机初始化
embedding_matrix[pad_idx] = np.zeros(embed_dim)  # <pad> 用全零向量

# 构建词->向量字典
print("Loading GloVe embeddings...")
glove_vocab = {}
with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == embed_dim + 1:
            word = parts[0]
            vec = np.array(parts[1:], dtype=float)
            glove_vocab[word] = vec
# 将词向量填入embedding_matrix
hit, miss = 0, 0
for word, idx in vocab.items():
    if word in glove_vocab:
        embedding_matrix[idx] = glove_vocab[word]
        hit += 1
    else:
        miss += 1
print(f"Total vocab size: {vocab_size}, Hit: {hit}, Miss: {miss}")

# 转换为tensor
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

class BiElmanAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, pad_idx=0):
        super(BiElmanAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # 初始化embedding层权重为预训练向量，设置为可训练
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = True  # 也可设False冻结词向量
        # Elman RNN层（双向）
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          bidirectional=True, batch_first=True, nonlinearity='tanh')
        # 层归一化：作用在双向RNN输出的隐藏状态上
        self.layernorm = nn.LayerNorm(hidden_dim * 2)
        # 注意力层：将隐藏向量映射为分数
        self.attn_fc = nn.Linear(hidden_dim * 2, 1)
        # 输出分类层
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) 的索引张量
        emb = self.embedding(x)               # (batch, seq_len, embed_dim)
        rnn_out, _ = self.rnn(emb)            # rnn_out: (batch, seq_len, 2*hidden_dim)
        norm_out = self.layernorm(rnn_out)    # 对最后一维做LayerNorm
        # 计算注意力权重
        # 首先通过tanh激活，再通过线性层得到注意力分数
        attn_scores = self.attn_fc(torch.tanh(norm_out)).squeeze(-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1)             # (batch, seq_len)
        # 用注意力权重加权求和RNN输出得到上下文向量
        # 将权重shape扩展为 (batch, seq_len, 1) 以便逐元素相乘
        attn_weights_expanded = attn_weights.unsqueeze(-1)           # (batch, seq_len, 1)
        context = (norm_out * attn_weights_expanded).sum(dim=1)      # (batch, 2*hidden_dim)
        # 最终分类输出
        logits = self.fc(context)            # (batch, num_classes)
        return logits, attn_weights

num_classes = 4
hidden_dim = 128
num_layers = 2
model = BiElmanAttentionModel(vocab_size, embed_dim, hidden_dim, num_classes, num_layers, pad_idx=pad_idx)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)
val_dataset = TextDataset(X_val, y_val) if X_val is not None else None

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

num_epochs = 5
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(1, num_epochs+1):
    model.train()
    epoch_loss = 0.0
    correct, total = 0, 0
    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}"):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits, attn_weights = model(batch_X) 
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_X.size(0)
    avg_loss = epoch_loss / total
    train_losses.append(avg_loss)
    train_acc = correct / total
    train_accs.append(train_acc)
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(model.state_dict(), f"checkpoints/pytorch_model_weights_epoch{epoch}.pth")
    if val_loader:
        model.eval()
        val_loss = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                logits, _ = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                preds = logits.argmax(dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_X.size(0)
        avg_val_loss = val_loss / total_val
        val_losses.append(avg_val_loss)
        val_acc = correct_val / total_val
        val_accs.append(val_acc)
        print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
    else:
        print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}")

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        logits, _ = model(batch_X)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1-score (macro): {test_f1:.4f}")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["World", "Sports", "Business", "Sci/Tech"]))

plt.figure(figsize=(6,4))
plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
if val_losses:
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(range(1, len(train_accs)+1), train_accs, label="Train Acc")
if val_accs:
    plt.plot(range(1, len(val_accs)+1), val_accs, label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png")
plt.close()

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, ["World","Sports","Business","Sci/Tech"], rotation=45)
plt.yticks(tick_marks, ["World","Sports","Business","Sci/Tech"])
thresh = cm.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

if len(test_tokens) > 0:
    sample_text = " ".join(test_tokens[0])
    sample_tokens = test_tokens[0]
    sample_ids = torch.tensor([pad_or_truncate(tokens_to_ids(sample_tokens))], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        _, attn_weights = model(sample_ids) 
        attn_weights = attn_weights.squeeze(0).cpu().numpy()  # shape: (seq_len,)
    real_len = min(len(sample_tokens), MAX_LEN)
    attn_weights = attn_weights[:real_len]
    words = sample_tokens[:real_len]
    plt.figure(figsize=(8,4))
    plt.title(f"Attention Weights for sample: \"{sample_text[:50]}...\"")
    x_pos = np.arange(len(words))
    plt.bar(x_pos, attn_weights, align='center')
    plt.xticks(x_pos, words, rotation=45)
    plt.ylabel('Attention Weight')
    plt.xlabel('Word')
    plt.tight_layout()
    plt.savefig("attention_weights_sample.png")
    plt.close()

print("All training and evaluation steps completed.")
