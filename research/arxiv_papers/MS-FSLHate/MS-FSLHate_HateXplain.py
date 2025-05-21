import json
import random
import re
import math
from collections import Counter

import numpy as np
import mindspore as ms
from mindspore import nn, context, Tensor, ops
import mindspore.dataset as ds
from tqdm import tqdm
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
# 通过多数投票确定最终标签
# ----------------------------
def compute_majority_label(annotators):
    labels = [ann.get("label", "").lower() for ann in annotators if "label" in ann]
    if not labels:
        return "normal"
    return Counter(labels).most_common(1)[0][0]

# ----------------------------
# 同义词替换（用于对抗数据增强）
# ----------------------------
def synonym_replacement(tokens, prob=0.1):
    new_tokens = tokens.copy()
    for i, word in enumerate(tokens):
        if random.random() < prob:
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.add(lemma.name())
            if synonyms:
                new_tokens[i] = random.choice(list(synonyms))
    return new_tokens

# ----------------------------
# 词汇表构建
# ----------------------------
def build_vocab(tokens_list, max_vocab_size=15000):
    special_tokens = ["<pad>", "<unk>", "@USER", "HASHTAG"]
    counts = {}
    for tokens in tqdm(tokens_list, desc="Building vocabulary"):
        for t in tokens:
            key = "HASHTAG" if t.startswith("HASHTAG_") else t
            counts[key] = counts.get(key, 0) + 1
    sorted_terms = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
    for term, _ in sorted_terms:
        if len(vocab) >= max_vocab_size:
            break
        if term not in vocab:
            vocab[term] = len(vocab)
    return vocab


# ----------------------------
def tokens_to_ids(tokens, vocab, max_length=128):
    ids = []
    for t in tokens:
        key = "HASHTAG" if t.startswith("HASHTAG_") else t
        ids.append(vocab.get(key, vocab["<unk>"]))
    if len(ids) < max_length:
        ids += [vocab["<pad>"]] * (max_length - len(ids))
    else:
        ids = ids[:max_length]
    return np.array(ids, dtype=np.int32)

# ----------------------------
# 从 JSON 文件加载数据
# ----------------------------
def load_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples, labels = [], []
    mapping = {"normal": 0, "offensive": 1, "hatespeech": 2}
    for sample in data:
        tokens = sample.get("post_tokens", [])
        annos = sample.get("annotators", [])
        lbl = compute_majority_label(annos)
        samples.append(tokens)
        labels.append(mapping.get(lbl, 0))
    return samples, np.array(labels, dtype=np.int32)

# ----------------------------
# 构建训练/测试数据集（含对抗增强）
# ----------------------------
def create_datasets(train_path, val_path, test_path,
                    max_length=128, batch_size=32, aug_prob=0.1):
    train_s, train_l = load_dataset(train_path)
    val_s,   val_l   = load_dataset(val_path)
    test_s,  test_l  = load_dataset(test_path)

    
    combined_s = train_s + val_s
    combined_l = np.concatenate([train_l, val_l], axis=0)

    def print_dist(name, arr):
        u, c = np.unique(arr, return_counts=True)
        print(f"{name} distribution:")
        for cls, cnt in zip(u, c):
            print(f"  Class {cls}: {cnt}")
        print()
    print_dist("Train+Val", combined_l)
    print_dist("Test", test_l)

    # 构建词表
    vocab = build_vocab(combined_s)

    # 对抗增强
    aug_s = [synonym_replacement(tokens, prob=aug_prob) for tokens in combined_s]
    combined_s += aug_s
    combined_l = np.concatenate([combined_l, combined_l], axis=0)

    # 转 ID
    def convert(samples):
        return np.array([
            tokens_to_ids(s, vocab, max_length)
            for s in tqdm(samples, desc="Converting samples")
        ])
    train_ids = convert(combined_s)
    test_ids  = convert(test_s)

    train_ds = ds.NumpySlicesDataset(
        {"data": train_ids, "label": combined_l},
        shuffle=True
    ).batch(batch_size, drop_remainder=True)
    test_ds = ds.NumpySlicesDataset(
        {"data": test_ids, "label": test_l},
        shuffle=False
    ).batch(batch_size, drop_remainder=True)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(combined_l), y=combined_l
    )
    return train_ds, test_ds, vocab, class_weights

# ----------------------------
class EnhancedClassifier(nn.Cell):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_classes, prompt_length, dropout_keep_prob):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.prompt_emb = ms.Parameter(
            ms.common.initializer.initializer(
                'normal', (prompt_length, embed_dim)
            )
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
        self.attn = nn.Dense(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm((hidden_dim * 2,))
        self.dropout = nn.Dropout(keep_prob=dropout_keep_prob)
        self.fc = nn.Dense(hidden_dim * 2, num_classes)

    def construct(self, x):
        bs = x.shape[0]
        emb = self.embedding(x)                  # [bs, seq, dim]
        prompt = self.prompt_emb.expand_dims(0) \
                 .tile((bs, 1, 1))              # [bs, P, dim]
        combo = ops.cat((prompt, emb), axis=1)  # [bs, P+seq, dim]

        c_in = combo.transpose(0, 2, 1)
        c_out = self.conv(c_in).transpose(0, 2, 1)
        lstm_out, _ = self.lstm(c_out)          # [bs, L, hid*2]

        weights = ops.softmax(self.attn(lstm_out), axis=1)
        ctx = ops.sum(lstm_out * weights, 1)    # [bs, hid*2]
        ctx = self.layer_norm(ctx)
        ctx = self.dropout(ctx)
        return self.fc(ctx)

# ----------------------------
# 模型评估函数
# ----------------------------
def evaluate_model(model, dataset):
    model.set_train(False)
    all_preds, all_labels, all_probs = [], [], []
    softmax_op = nn.Softmax()
    with tqdm(dataset.create_dict_iterator(), desc="Evaluating") as pbar:
        for batch in pbar:
            x = batch["data"]
            y = batch["label"].asnumpy()
            logits = model(x)
            probs = softmax_op(logits).asnumpy()
            preds = ops.Argmax(axis=1)(logits).asnumpy()
            all_preds.extend(preds)
            all_labels.extend(y)
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)

    report_str = classification_report(
        all_labels, all_preds,
        target_names=["normal","offensive","hatespeech"],
        digits=4, zero_division=0
    )
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=["normal","offensive","hatespeech"],
        output_dict=True, digits=4, zero_division=0
    )
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    bin_labels = label_binarize(all_labels, classes=[0,1,2])
    roc_auc = roc_auc_score(bin_labels, all_probs, multi_class='ovr')
    return weighted_f1, roc_auc, report_str

# ----------------------------
# 自定义训练步（含梯度裁剪）
# ----------------------------
class CustomTrainOneStep(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, clip_norm):
        super().__init__(network, optimizer)
        self.clip_norm = clip_norm
        self.grad_op = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad_op(self.network, self.weights)(*inputs)
        grads = ops.clip_by_global_norm(grads, self.clip_norm)
        return ops.depend(loss, self.optimizer(grads))

# ----------------------------
# 主流程
# ----------------------------
def main():
    config = {
        # 数据路径 & 批大小
        "train_path":       "MS-FSLHate/train_full.json",
        "val_path":         "MS-FSLHate/val_full.json",
        "test_path":        "MS-FSLHate/test_full.json",
        "batch_size":       32,
        "max_length":       128,

        # 模型尺寸
        "embed_dim":        300,
        "hidden_dim":       256,
        "prompt_length":    10,

        # 正则化
        "dropout_keep_prob":0.7,
        "weight_decay":     1e-5,

        # 优化器 & 学习率
        "learning_rate":    5e-4,
        "min_lr":           1e-5,

        # 训练细节
        "num_epochs":       3,
        "grad_clip":        1.0,

        # 数据增强
        "aug_prob":         0.1,
    }

    # 加载数据集
    train_ds, test_ds, vocab, class_weights = create_datasets(
        config["train_path"], config["val_path"], config["test_path"],
        config["max_length"], config["batch_size"],
        config["aug_prob"]
    )
    print("Vocab size:", len(vocab))

    # 构建模型
    model = EnhancedClassifier(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=3,
        prompt_length=config["prompt_length"],
        dropout_keep_prob=config["dropout_keep_prob"]
    )

    # 计算训练步数
    train_batches = sum(1 for _ in train_ds.create_dict_iterator())
    total_steps   = train_batches * config["num_epochs"]
    print(f"Total training steps: {total_steps}")

    # 学习率调度
    lr_schedule = nn.cosine_decay_lr(
        min_lr=config["min_lr"],
        max_lr=config["learning_rate"],
        total_step=total_steps,
        step_per_epoch=train_batches,
        decay_epoch=config["num_epochs"]
    )

    # 损失函数 & 优化器
    loss_fn = nn.CrossEntropyLoss(
        weight=Tensor(class_weights, ms.float32),
        reduction="mean"
    )
    optimizer = nn.AdamWeightDecay(
        model.trainable_params(),
        learning_rate=lr_schedule,
        weight_decay=config["weight_decay"]
    )

    train_net = CustomTrainOneStep(
        nn.WithLossCell(model, loss_fn),
        optimizer,
        config["grad_clip"]
    )

    # 训练 & 评估
    for epoch in range(config["num_epochs"]):
        model.set_train()
        running_loss = 0.0
        with tqdm(train_ds.create_dict_iterator(), total=train_batches,
                  desc=f"Training Epoch {epoch+1}/{config['num_epochs']}") as pbar:
            for i, batch in enumerate(pbar, 1):
                loss = train_net(batch["data"], batch["label"])
                running_loss += loss.asnumpy()
                avg_loss = running_loss / i
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

        f1, auc, rpt = evaluate_model(model, test_ds)
        print(f"After Epoch {epoch+1} → Weighted F1={f1:.4f}, ROC AUC={auc:.4f}")
        print("Classification Report:\n", rpt)

    # 保存最终模型
    ms.save_checkpoint(model, "final_model.ckpt")
    print("Experiment completed.")

if __name__ == "__main__":
    main()
