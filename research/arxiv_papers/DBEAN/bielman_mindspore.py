import re
import numpy as np
import random
from collections import Counter
from nltk.corpus import wordnet, stopwords
import nltk
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, dtype as mstype, save_checkpoint
from mindspore.dataset import GeneratorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import mindspore.dataset as ds

import os

nltk.download('stopwords')
nltk.download('wordnet')

random.seed(42)
np.random.seed(42)

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

train_csv_path = "train.csv"
test_csv_path = "test.csv"

column_names = ["label", "title", "description"]

train_dataset = ds.CSVDataset(train_csv_path, column_names=column_names, shuffle=False)
test_dataset = ds.CSVDataset(test_csv_path, column_names=column_names, shuffle=False)

def preprocess_func(label, title, description):
    if not isinstance(title, str):
        try:
            title = title.decode("utf-8")
        except Exception:
            title = str(title)
    if not isinstance(description, str):
        try:
            description = description.decode("utf-8")
        except Exception:
            description = str(description)
    text = title + " " + description
    label = int(label) - 1
    return text, label

train_dataset = train_dataset.map(operations=preprocess_func,
                                  input_columns=["label", "title", "description"],
                                  output_columns=["text", "label"])
test_dataset = test_dataset.map(operations=preprocess_func,
                                input_columns=["label", "title", "description"],
                                output_columns=["text", "label"])

train_dataset = train_dataset.project(["text", "label"])
test_dataset = test_dataset.project(["text", "label"])

train_dataset, val_dataset = train_dataset.split([0.9, 0.1])

def dataset_to_list(ds_obj):
    texts = []
    labels = []
    for data in ds_obj.create_dict_iterator(num_epochs=1, output_numpy=True):
        txt = data["text"]
        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        elif isinstance(txt, np.ndarray):
            txt = txt.item() if txt.size == 1 else str(txt)
        texts.append(txt)
        labels.append(int(data["label"]))
    return texts, labels

train_texts, train_labels = dataset_to_list(train_dataset)
val_texts, val_labels = dataset_to_list(val_dataset)
test_texts, test_labels = dataset_to_list(test_dataset)

print("训练集大小:", len(train_texts))
print("验证集大小:", len(val_texts))
print("测试集大小:", len(test_texts))

stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return words

train_tokens = [preprocess_text(txt) for txt in tqdm(train_texts, desc="Preprocessing Train Texts")]
val_tokens = [preprocess_text(txt) for txt in tqdm(val_texts, desc="Preprocessing Val Texts")] if val_texts else []
test_tokens = [preprocess_text(txt) for txt in tqdm(test_texts, desc="Preprocessing Test Texts")]

def synonym_replace(words, n=1):
    new_words = words.copy()
    available_indices = [i for i, w in enumerate(new_words) if w not in stop_words]
    random.shuffle(available_indices)
    num_replaced = 0
    for idx in available_indices:
        syns = wordnet.synsets(new_words[idx])
        if syns:
            syn_lemmas = [lemma.name() for lemma in syns[0].lemmas() if lemma.name().lower() != new_words[idx]]
            if syn_lemmas:
                new_words[idx] = syn_lemmas[0].replace('_', ' ')
                num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

def random_deletion(words, p=0.2):
    if len(words) == 0:
        return words
    new_words = []
    for w in words:
        if w in stop_words:
            new_words.append(w)
        else:
            if random.random() > p:
                new_words.append(w)
    if len(new_words) == 0:
        new_words.append(random.choice(words))
    return new_words

aug_train_tokens = []
aug_train_labels = []
for tokens, label in tqdm(zip(train_tokens, train_labels), total=len(train_tokens), desc="Data Augmentation"):
    aug_train_tokens.append(tokens)
    aug_train_labels.append(label)
    if len(tokens) > 0:
        aug_type = random.choice(["sr", "rd"])
        new_tokens = synonym_replace(tokens) if aug_type == "sr" else random_deletion(tokens, p=0.1)
        aug_train_tokens.append(new_tokens)
        aug_train_labels.append(label)

train_tokens = aug_train_tokens
train_labels = aug_train_labels

word_counter = Counter()
for tokens in train_tokens:
    word_counter.update(tokens)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for word, freq in word_counter.items():
    vocab[word] = len(vocab)
vocab_size = len(vocab)
pad_idx = vocab[PAD_TOKEN]
unk_idx = vocab[UNK_TOKEN]

def tokens_to_ids(tokens):
    return [vocab.get(w, unk_idx) for w in tokens]

MAX_LEN = 100
def pad_or_truncate(seq, max_len=MAX_LEN):
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        return seq + [pad_idx] * (max_len - len(seq))

train_ids = [pad_or_truncate(tokens_to_ids(seq)) for seq in tqdm(train_tokens, desc="Converting Train Tokens")]
val_ids = [pad_or_truncate(tokens_to_ids(seq)) for seq in tqdm(val_tokens, desc="Converting Val Tokens")] if val_tokens else []
test_ids = [pad_or_truncate(tokens_to_ids(seq)) for seq in tqdm(test_tokens, desc="Converting Test Tokens")]

X_train = np.array(train_ids, dtype=np.int32)
y_train = np.array(train_labels, dtype=np.int32)
X_val = np.array(val_ids, dtype=np.int32) if val_tokens else None
y_val = np.array(val_labels, dtype=np.int32) if val_tokens else None
X_test = np.array(test_ids, dtype=np.int32)
y_test = np.array(test_labels, dtype=np.int32)

embed_dim = 100
glove_path = "glove.6B.100d.txt"
embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embed_dim)).astype(np.float32)
embedding_matrix[pad_idx] = np.zeros(embed_dim, dtype=np.float32)

print("Loading GloVe embeddings...")
glove_vocab = {}
with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == embed_dim + 1:
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove_vocab[word] = vec
hit, miss = 0, 0
for word, idx in vocab.items():
    if word in glove_vocab:
        embedding_matrix[idx] = glove_vocab[word]
        hit += 1
    else:
        miss += 1
print(f"Vocab size: {vocab_size}, Hit: {hit}, Miss: {miss}")

embedding_matrix_ms = Tensor(embedding_matrix, mstype.float32)

class BiElmanAttentionNet(nn.Cell):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, pad_idx=0):
        super(BiElmanAttentionNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, embedding_table=embedding_matrix_ms, padding_idx=pad_idx)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          has_bias=True, bidirectional=True, batch_first=True, nonlinearity='tanh')
        self.layernorm = nn.LayerNorm([hidden_dim * 2])
        self.attn_fc = nn.Dense(hidden_dim * 2, 1)
        self.fc = nn.Dense(hidden_dim * 2, num_classes)
        self.softmax = ops.Softmax(axis=1)

    def construct(self, x):
        emb = self.embedding(x)
        rnn_out, _ = self.rnn(emb)
        norm_out = self.layernorm(rnn_out)
        attn_scores = self.attn_fc(ops.tanh(norm_out)).squeeze(-1)
        attn_weights = self.softmax(attn_scores)
        attn_weights_expanded = ops.ExpandDims()(attn_weights, -1)
        context = ops.ReduceSum()(norm_out * attn_weights_expanded, 1)
        logits = self.fc(context)
        return logits, attn_weights

hidden_dim = 128
num_layers = 2
num_classes = 4
model = BiElmanAttentionNet(vocab_size, embed_dim, hidden_dim, num_classes, num_layers, pad_idx=pad_idx)
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

train_dataset_gen = GeneratorDataset(list(zip(X_train, y_train)), column_names=["text", "label"], shuffle=True)
test_dataset_gen = GeneratorDataset(list(zip(X_test, y_test)), column_names=["text", "label"], shuffle=False)
val_dataset_gen = GeneratorDataset(list(zip(X_val, y_val)), column_names=["text", "label"], shuffle=False) if X_val is not None else None

batch_size = 64
train_dataset_gen = train_dataset_gen.batch(batch_size)
test_dataset_gen = test_dataset_gen.batch(batch_size)
if val_dataset_gen:
    val_dataset_gen = val_dataset_gen.batch(batch_size)

def train_one_epoch(model, dataset, epoch):
    model.set_train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    for data in tqdm(dataset.create_tuple_iterator(), desc=f"Epoch {epoch} Training"):
        texts, labels = data
        logits, _ = model(texts)
        loss = loss_fn(logits, labels)
        grads = ops.GradOperation(get_by_list=True)(lambda x, y: loss_fn(model(x)[0], y), model.trainable_params())(texts, labels)
        optimizer(grads)
        batch_size_local = texts.shape[0]
        total_loss += float(loss.asnumpy()) * batch_size_local
        total_samples += batch_size_local
        pred = np.argmax(logits.asnumpy(), axis=1)
        correct += np.sum(pred == labels.asnumpy())
    avg_loss = total_loss / total_samples
    acc = correct / total_samples
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    save_checkpoint(model, f"checkpoints/mindspore_model_weights_{epoch}.ckpt")
    return avg_loss, acc

def evaluate(model, dataset):
    model.set_train(False)
    total_loss = 0.0
    total_samples = 0
    correct = 0
    all_preds = []
    all_labels = []
    for data in tqdm(dataset.create_tuple_iterator(), desc="Evaluating"):
        texts, labels = data
        logits, _ = model(texts)
        loss = loss_fn(logits, labels)
        batch_size_local = texts.shape[0]
        total_loss += float(loss.asnumpy()) * batch_size_local
        total_samples += batch_size_local
        preds = np.argmax(logits.asnumpy(), axis=1)
        correct += np.sum(preds == labels.asnumpy())
        all_preds.extend(list(preds))
        all_labels.extend(list(labels.asnumpy()))
    avg_loss = total_loss / total_samples
    acc = correct / total_samples
    return avg_loss, acc, all_preds, all_labels

num_epochs = 5
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(1, num_epochs+1):
    train_loss, train_acc = train_one_epoch(model, train_dataset_gen, epoch)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    if val_dataset_gen:
        val_loss, val_acc, _, _ = evaluate(model, val_dataset_gen)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    else:
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

test_loss, test_acc, test_preds, test_labels_eval = evaluate(model, test_dataset_gen)
test_f1 = f1_score(test_labels_eval, test_preds, average='macro')
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1-score (macro): {test_f1:.4f}")
print("Classification Report:")
print(classification_report(test_labels_eval, test_preds, target_names=["World","Sports","Business","Sci/Tech"]))

plt.figure()
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
if val_losses:
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("ms_loss_curve.png")
plt.close()

plt.figure()
plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
if val_accs:
    plt.plot(range(1, len(val_accs)+1), val_accs, label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.tight_layout()
plt.savefig("ms_accuracy_curve.png")
plt.close()

cm = confusion_matrix(test_labels_eval, test_preds)
plt.figure()
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, ["World","Sports","Business","Sci/Tech"], rotation=45)
plt.yticks(tick_marks, ["World","Sports","Business","Sci/Tech"])
thresh = cm.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j] > thresh else "black")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig("ms_confusion_matrix.png")
plt.close()

if len(test_tokens) > 0:
    sample_tokens = test_tokens[0]
    sample_ids = pad_or_truncate(tokens_to_ids(sample_tokens))
    sample_tensor = Tensor(np.array([sample_ids], dtype=np.int32))
    model.set_train(False)
    _, attn_weights_ms = model(sample_tensor)
    attn_weights = attn_weights_ms.asnumpy().squeeze(0)
    real_len = min(len(sample_tokens), MAX_LEN)
    attn_weights = attn_weights[:real_len]
    words = sample_tokens[:real_len]
    plt.figure(figsize=(8,4))
    plt.title("Attention Weights for Sample")
    x_pos = np.arange(len(words))
    plt.bar(x_pos, attn_weights, align='center')
    plt.xticks(x_pos, words, rotation=45)
    plt.ylabel('Attention Weight')
    plt.xlabel('Word')
    plt.tight_layout()
    plt.savefig("ms_attention_weights_sample.png")
    plt.close()

print("Training and evaluation completed (MindSpore).")
