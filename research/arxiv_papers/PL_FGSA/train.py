import os, json, csv
from collections import Counter
from tqdm import tqdm
from mindspore import nn, ops, Tensor, context
from mindspore.dataset import GeneratorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

context.set_context(mode=context.PYNATIVE_MODE)

class TextCNN(nn.Cell):
    def __init__(self, vocab_size, embed_dim=128, num_classes=3, kernel_sizes=(3,4,5), num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.CellList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.relu = nn.ReLU()
        self.max_pool = ops.ReduceMax(keep_dims=False)
        self.concat = ops.Concat(axis=1)
        self.dropout = nn.Dropout(p=0.7)
        self.fc = nn.Dense(len(kernel_sizes)*num_filters, num_classes)

    def construct(self, x):
        x = self.embedding(x).transpose(0, 2, 1)
        conv_outs = [self.max_pool(self.relu(conv(x)), 2) for conv in self.convs]
        out = self.concat(conv_outs)
        out = self.dropout(out)
        return self.fc(out)

def build_vocab(json_path, save_path, min_freq=1):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    counter = Counter()
    for x in data:
        if x.get("label") is not None:
            words = f"{x['text']} [ASP] {x['aspect']}".lower().split()
            counter.update(words)
    word2id = {w: i+1 for i, (w, c) in enumerate(counter.items()) if c >= min_freq}
    word2id['[PAD]'] = 0
    json.dump(word2id, open(save_path, 'w', encoding='utf-8'), indent=2)
    return word2id

def load_tokenizer(vocab_path):
    word2id = json.load(open(vocab_path, encoding='utf-8'))
    return lambda text: [word2id.get(w, 0) for w in text.lower().split()], len(word2id)

def load_dataset(json_path, tokenizer, max_len=128):
    data = [x for x in json.load(open(json_path, encoding='utf-8')) if x.get("label") is not None]
    def encode(x):
        text = f"{x['text']} [ASP] {x['aspect']}"
        ids = tokenizer(text)[:max_len] + [0] * max(0, max_len - len(tokenizer(text)))
        return np.array(ids, np.int32), np.array(x['label'], np.int32)
    return [encode(x) for x in data]

def make_dataset(data, batch_size):
    def gen(): yield from data
    return GeneratorDataset(gen, ["input_ids", "label"], shuffle=True).batch(batch_size)

def train_eval(dataset_name, batch_size=32, epochs=10):
    json_path_map = {
        'semeval': "processed/semeval/laptops/Laptop_Train_v2.json",
        'ecare': "processed/ecare/train_full.json",
        'mams': "processed/mams/train.json",
        'sst2': "processed/sst2/train.json"
    }
    val_path_map = {
        'mams': "processed/mams/val.json"
    }

    json_path = json_path_map[dataset_name]
    vocab_path = f"vocab_{dataset_name}.json"
    log_file = f"log_{dataset_name}.txt"
    
    if not os.path.exists(vocab_path):
        build_vocab(json_path, vocab_path)
    tokenizer, vocab_size = load_tokenizer(vocab_path)

    train_data = load_dataset(json_path, tokenizer)
    if dataset_name == 'mams':
        val_data = load_dataset(val_path_map['mams'], tokenizer)
    else:
        train_data, tmp = train_test_split(train_data, test_size=0.2, random_state=42)
        val_data, _ = train_test_split(tmp, test_size=0.5)

    train_ds = make_dataset(train_data, batch_size)
    val_ds = make_dataset(val_data, batch_size)

    model = TextCNN(vocab_size)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    optim = nn.Adam(model.trainable_params(), learning_rate=1e-3, weight_decay=1e-4)
    net = nn.TrainOneStepCell(nn.WithLossCell(model, loss_fn), optim)

    log = open(log_file, 'w', encoding='utf-8')
    for epoch in range(epochs):
        model.set_train()
        preds, labels = [], []
        pbar = tqdm(train_ds.create_tuple_iterator(), desc=f"{dataset_name.upper()} Epoch {epoch+1}")
        for x, y in pbar:
            loss = net(x, y)
            loss_np = loss.asnumpy()
            loss_scalar = loss_np.mean() if hasattr(loss_np, 'mean') else float(loss_np)
            pbar.set_postfix(loss=loss_scalar)

            logits = model(x).asnumpy()
            preds.extend(np.argmax(logits, axis=1))
            labels.extend(y.asnumpy())

        acc = accuracy_score(labels, preds)
        try:
            _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        except:
            f1 = 0.0
        log.write(f"[Epoch {epoch+1}] acc={acc:.4f}, f1={f1:.4f}\n")

    model.set_train(False)
    y_true, y_pred = [], []
    for x, y in val_ds.create_tuple_iterator():
        logits = model(x).asnumpy()
        y_pred.extend(np.argmax(logits, axis=1))
        y_true.extend(y.asnumpy())
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    log.write(f"VAL acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}\n")
    log.close()

    return {'dataset': dataset_name, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}


if __name__ == '__main__':
    results = []
    for dataset in [
                    # 'ecare',
                    # 'semeval', 
                    # 'mams', 
                    'sst2'
                    ]:
        result = train_eval(dataset, batch_size=32, epochs=5)
        results.append(result)

    with open("results_summary.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "acc", "prec", "rec", "f1"])
        writer.writeheader()
        writer.writerows(results)
