import random
import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import nn, context, Tensor, ops
import mindspore.dataset as ds
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from nltk.corpus import wordnet
import nltk

# 如果需要下载 wordnet 数据，请取消下一行注释并运行一次
# nltk.download('wordnet', download_dir='/mnt/workspace/nltk_data')
nltk.data.path.append('/mnt/workspace/nltk_data')

# 设置 MindSpore 运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# ----------------------------
# 对抗数据增强：同义词替换
# ----------------------------
def synonym_replacement(tokens, prob=0.1):
    new_tokens = tokens.copy()
    for i, w in enumerate(tokens):
        if random.random() < prob:
            syns = set()
            for syn in wordnet.synsets(w):
                for lemma in syn.lemmas():
                    if lemma.name() != w:
                        syns.add(lemma.name())
            if syns:
                new_tokens[i] = random.choice(list(syns))
    return new_tokens

# ----------------------------
# 构建词表
# ----------------------------
def build_vocab(tokens_list, max_size=20000):
    special = ["<pad>", "<unk>", "@USER", "HASHTAG"]
    counts = {}
    for tokens in tqdm(tokens_list, desc="Building vocab"):
        for t in tokens:
            key = "HASHTAG" if t.startswith("HASHTAG_") else t
            counts[key] = counts.get(key, 0) + 1
    sorted_terms = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {tok: idx for idx, tok in enumerate(special)}
    for term, _ in sorted_terms:
        if len(vocab) >= max_size:
            break
        if term not in vocab:
            vocab[term] = len(vocab)
    return vocab

# ----------------------------
# tokens → fixed-length IDs
# ----------------------------
def tokens_to_ids(tokens, vocab, max_len=128):
    ids = []
    for t in tokens:
        key = "HASHTAG" if t.startswith("HASHTAG_") else t
        ids.append(vocab.get(key, vocab["<unk>"]))
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return np.array(ids, dtype=np.int32)

# ----------------------------
# 读取 CSV，分层划分，构建 MindSpore 数据集
# ----------------------------
def create_datasets(csv_path, test_size, max_len, batch_size):
    df = pd.read_csv(csv_path).rename(columns={'tweet':'text'})
    df['tokens'] = df['text'].str.split()
    X, y = df['tokens'].tolist(), df['class'].to_numpy().astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    print("Train distribution:")
    for cls, cnt in zip(*np.unique(y_train, return_counts=True)):
        print(f"  Class {cls}: {cnt}")
    print("Test distribution:")
    for cls, cnt in zip(*np.unique(y_test, return_counts=True)):
        print(f"  Class {cls}: {cnt}")
    print()

    vocab = build_vocab(X_train)

    # 对抗增强
    X_train_aug = [synonym_replacement(tokens, prob=0.1) for tokens in X_train]
    X_train_all = X_train + X_train_aug
    y_train_all = np.concatenate([y_train, y_train], axis=0)

    def to_ids(data):
        return np.array([
            tokens_to_ids(tokens, vocab, max_len)
            for tokens in tqdm(data, desc="Converting to IDs")
        ])
    train_ids = to_ids(X_train_all)
    test_ids  = to_ids(X_test)

    train_ds = ds.NumpySlicesDataset(
        {"data": train_ids, "label": y_train_all},
        shuffle=True
    ).batch(batch_size, drop_remainder=True)

    test_ds  = ds.NumpySlicesDataset(
        {"data": test_ids, "label": y_test},
        shuffle=False
    ).batch(batch_size, drop_remainder=True)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train_all), y=y_train_all
    )
    return train_ds, test_ds, vocab, class_weights

# ----------------------------
# 模型

class EnhancedClassifier(nn.Cell):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_classes, prompt_len, dropout_prob):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.prompt_emb = ms.Parameter(
            ms.common.initializer.initializer('normal', (prompt_len, embed_dim))
        )
        self.conv = nn.SequentialCell([ 
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ])
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=hidden_dim,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=0.2
        )
        self.attn = nn.Dense(hidden_dim*2, 1)
        self.layer_norm = nn.LayerNorm((hidden_dim*2,))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Dense(hidden_dim*2, num_classes)

    def construct(self, x):
        emb = self.embedding(x)                     # [bs, seq, dim]
        bs = x.shape[0]
        prompt = self.prompt_emb.expand_dims(0) \
                 .tile((bs, 1, 1))                 # [bs, P, dim]
        combo = ops.cat((prompt, emb), axis=1)     # [bs, P+seq, dim]

        c_in = combo.transpose(0,2,1)
        c_out = self.conv(c_in).transpose(0,2,1)
        lstm_out, _ = self.lstm(c_out)             # [bs, L, hid*2]

        weights = ops.softmax(self.attn(lstm_out), axis=1)
        ctx = ops.sum(lstm_out * weights, 1)
        ctx = self.layer_norm(ctx)
        ctx = self.dropout(ctx)
        return self.fc(ctx)

# ----------------------------
# 自定义训练步（梯度裁剪）
# ----------------------------
class CustomTrainOneStep(nn.TrainOneStepCell):
    def __init__(self, net, optimizer, clip_norm):
        super().__init__(net, optimizer)
        self.clip_norm = clip_norm
        self.grad_op   = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss  = self.network(*inputs)
        grads = self.grad_op(self.network, self.weights)(*inputs)
        grads = ops.clip_by_global_norm(grads, self.clip_norm)
        return ops.depend(loss, self.optimizer(grads))

# ----------------------------
# 评估函数
# ----------------------------
def evaluate_model(model, dataset, desc="Evaluating"):
    model.set_train(False)
    all_preds, all_labels, all_probs = [], [], []
    softmax = nn.Softmax()
    with tqdm(dataset.create_dict_iterator(), desc=desc) as pbar:
        for batch in pbar:
            logits = model(batch["data"])
            probs  = softmax(logits).asnumpy()
            preds  = ops.Argmax(axis=1)(logits).asnumpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].asnumpy())
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)

    report_str = classification_report(
        all_labels, all_preds,
        target_names=["neither","offensive","hate-speech"],
        digits=4, zero_division=0
    )
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=["neither","offensive","hate-speech"],
        output_dict=True, digits=4, zero_division=0
    )
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    bin_labels = label_binarize(all_labels, classes=[0,1,2])
    roc_auc    = roc_auc_score(bin_labels, all_probs, multi_class='ovr')
    return weighted_f1, roc_auc, report_str

# ----------------------------
# 主流程
# ----------------------------
def main():
    cfg = {
        "csv_path":     "labeled_data.csv",
        "test_size":    0.2,
        "max_len":      128,
        "batch_size":   32,
        "embed_dim":    300,
        "hidden_dim":   256,
        "num_classes":  3,
        "prompt_len":   10,
        "dropout_prob": 0.3,    
        "learning_rate":5e-4,
        "min_lr":       1e-5,
        "weight_decay": 1e-5,
        "num_epochs":   10,
        "grad_clip":    1.0
    }

    # 加载数据集
    train_ds, test_ds, vocab, class_weights = create_datasets(
        cfg["csv_path"],
        test_size=cfg["test_size"],
        max_len=cfg["max_len"],
        batch_size=cfg["batch_size"]
    )
    print("Vocab size:", len(vocab))

    # 构建模型
    model = EnhancedClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"],
        prompt_len=cfg["prompt_len"],
        dropout_prob=cfg["dropout_prob"]
    )

    # 计算训练步数
    train_batches = sum(1 for _ in train_ds.create_dict_iterator())
    total_steps   = train_batches * cfg["num_epochs"]
    print(f"Total training steps: {total_steps}")

    # 学习率调度
    lr_schedule = nn.cosine_decay_lr(
        min_lr=cfg["min_lr"],
        max_lr=cfg["learning_rate"],
        total_step=total_steps,
        step_per_epoch=train_batches,
        decay_epoch=cfg["num_epochs"]
    )

    # 损失和优化器
    loss_fn = nn.CrossEntropyLoss(
        weight=Tensor(class_weights, ms.float32),
        reduction="mean"
    )
    optimizer = nn.AdamWeightDecay(
        model.trainable_params(),
        learning_rate=lr_schedule,
        weight_decay=cfg["weight_decay"]
    )
    train_net = CustomTrainOneStep(nn.WithLossCell(model, loss_fn),
                                   optimizer,
                                   cfg["grad_clip"])

    # 训练 & 评估循环
    for epoch in range(cfg["num_epochs"]):
        model.set_train(True)
        running_loss = 0.0
        with tqdm(train_ds.create_dict_iterator(), total=train_batches,
                  desc=f"Epoch {epoch+1}/{cfg['num_epochs']}") as pbar:
            for i, batch in enumerate(pbar, 1):
                loss = train_net(batch["data"], batch["label"])
                running_loss += loss.asnumpy()
                avg = running_loss / i
                pbar.set_postfix(loss=f"{avg:.4f}")

        f1, auc, rpt = evaluate_model(model, test_ds, desc=f"Evaluating Epoch {epoch+1}")
        print(f"\nAfter Epoch {epoch+1}: Weighted F1={f1:.4f}, ROC AUC={auc:.4f}")
        print(rpt)

    # 保存模型
    ms.save_checkpoint(model, "final_model.ckpt")

if __name__ == "__main__":
    main()
