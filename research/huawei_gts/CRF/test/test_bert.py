import os
import random

import mindspore
import mindspore.dataset as ds
import numpy as np
from ..model.lstm_crf_model import CRF
from mindnlp.engine import Trainer
from mindnlp.transformers import AutoModel, AutoTokenizer
from mindspore import nn
from mindspore.nn import AdamWeightDecay as AdamW
from tqdm.autonotebook import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)
    mindspore.dataset.config.set_seed(seed)


# 读取文本，返回词典，索引表，句子，标签
def read_data(path):
    sentences = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        sent = []
        label = []
        for line in f:
            parts = line.split()
            if len(parts) == 0:
                if len(sent) != 0:
                    sentences.append(sent)
                    labels.append(label)
                sent = []
                label = []
            else:
                sent.append(parts[0])
                label.append(parts[-1])

    return (sentences, labels)


def read_vocab(path):
    vocab_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for word in f:
            vocab_list.append(word.strip())
    return vocab_list


def get_entity(decode):
    starting = False
    p_ans = []
    for i, label in enumerate(decode):
        if label > 0:
            if label % 2 == 1:
                starting = True
                p_ans.append(([i], labels_text_mp[label // 2]))
            elif starting:
                p_ans[-1][0].append(i)
        else:
            starting = False
    return p_ans


# 处理数据
class Feature(object):
    def __init__(self, sent, label):
        self.sent = sent
        label = [LABEL_MAP[c] for c in label]
        self.token_ids = list(tokenizer(' '.join(sent)).input_ids)
        self.seq_length = len(self.token_ids) if len(self.token_ids) - 2 < max_Len else max_Len + 2
        offset = tokenizer(' '.join(sent), return_offsets_mapping=True).offset_mapping
        self.labels = self.get_labels(offset, label)
        self.labels = [0] + self.labels[:max_Len] + [0]
        self.labels = self.labels + [0] * (max_Len - len(self.labels) + 2)

        self.token_ids = [101] + self.token_ids[1:-1][:max_Len] + [102]
        self.token_ids = self.token_ids + [0] * (max_Len - len(self.token_ids) + 2)
        self.entity = get_entity(self.labels)

    def get_labels(self, offset_mapping, label):
        sent_len, count, index = 0, 0, 0
        label_new = []
        for l, r in offset_mapping:
            if l != 0 or r != 0:
                if count == sent_len:
                    sent_len += len(self.sent[index])
                    index += 1
                count += r - l
                label_new.append(label[index - 1])

        return label_new


class GetDatasetGenerator:
    def __init__(self, path):
        data = read_data(path)
        self.features = [Feature(data[0][i], data[1][i]) for i in range(len(data[0]))]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        token_ids = feature.token_ids
        labels = feature.labels

        return token_ids, feature.seq_length, labels


def process_dataset(source, batch_size, shuffle):
    dataset = ds.GeneratorDataset(source, ["ids", "seq_length", "labels"], shuffle=shuffle)
    dataset = dataset.batch(batch_size=batch_size)

    return dataset


def debug_dataset(dataset):
    dataset = dataset.batch(batch_size=16)
    for data in dataset.create_dict_iterator():
        print(data["data"].shape, data["label"].shape)
        break


def get_metric(P_ans, valid):
    predict_score = 0  # 预测正确个数
    predict_number = 0  # 预测结果个数
    totol_number = 0  # 标签个数
    for i in range(len(P_ans)):
        predict_number += len(P_ans[i])
        totol_number += len(valid.features[i].entity)
        pred_true = [x for x in valid.features[i].entity if x in P_ans[i]]
        predict_score += len(pred_true)
    P = predict_score / predict_number if predict_number > 0 else 0.
    R = predict_score / totol_number if totol_number > 0 else 0.
    f1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0.
    print(f'f1 = {f1}， P(准确率) = {P}, R(召回率) = {R}')


def get_optimizer(model):
    param_optimizer = list(model.parameters_and_names())

    no_decay = ['bias', 'layer_norm.bias', 'layer_norm.weight']
    crf_p = [n for n, p in param_optimizer if str(n).find('crf') != -1]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and n not in crf_p],
         'weight_decay': 0.8},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and n not in crf_p],
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if n in crf_p], 'lr': 3e-3, 'weight_decay': 0.8},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, learning_rate=3e-5, eps=1e-8)  # 学习率不宜过大，不然预测结果可能都是0

    return optimizer


class Bert_LSTM_CRF(nn.Cell):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert_model.config.hidden_size
        self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.crf_hidden_fc = nn.Dense(hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True, reduction='sum')

    def construct(self, ids, seq_length=None, labels=None):
        attention_mask = (ids > mindspore.tensor(0))
        output = self.bert_model(input_ids=ids, attention_mask=attention_mask)
        lstm_feat, _ = self.bilstm(output[0])
        emissions = self.crf_hidden_fc(lstm_feat)
        loss_crf = self.crf(emissions, tags=labels, seq_length=seq_length)

        return loss_crf


seed_everything(42)
max_Len = 113
Entity = ['PER', 'LOC', 'ORG', 'MISC']
labels_text_mp = {k: v for k, v in enumerate(Entity)}
LABEL_MAP = {'O': 0}
for i, e in enumerate(Entity):
    LABEL_MAP[f'B-{e}'] = 2 * (i + 1) - 1
    LABEL_MAP[f'I-{e}'] = 2 * (i + 1)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
train = GetDatasetGenerator('../../../../huawei_gts/CRF/conll2003/train.txt')
test = GetDatasetGenerator('../../../../huawei_gts/CRF/conll2003/test.txt')
dev = GetDatasetGenerator('../../../../huawei_gts/CRF/conll2003/valid.txt')

epochs = 3
batch_size = 16
dataset_train = process_dataset(train, batch_size=batch_size, shuffle=False)
model = Bert_LSTM_CRF(num_labels=len(Entity) * 2 + 1)
optimizer = get_optimizer(model)
trainer = Trainer(network=model, train_dataset=dataset_train, optimizer=optimizer, epochs=epochs)

trainer.run(tgt_columns="labels")

# 预测：train
dataset_train = process_dataset(train, batch_size=batch_size, shuffle=False)
size = dataset_train.get_dataset_size()
steps = size
decodes = []
model.set_train(False)
with tqdm(total=steps) as t:
    for batch, (token_ids, seq_length, labels) in enumerate(dataset_train.create_tuple_iterator()):
        score, history = model(token_ids, seq_length=seq_length)
        best_tags = model.crf.post_decode(score, history, seq_length)
        decode = [[y.asnumpy().item() for y in x] for x in best_tags]
        decodes.extend(list(decode))
        t.update(1)

v_pred = [get_entity(x) for x in decodes]
get_metric(v_pred, train)
