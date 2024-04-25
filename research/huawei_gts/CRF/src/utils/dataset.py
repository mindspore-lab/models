"""dataset util"""

import mindspore.dataset as ds


# 返回词典映射表(单词，数量)、词数字典（单词，字典编号）
def get_dict(sentences):
    max_number = 1
    char_number_dict = {}

    id_indexs = {}
    id_indexs[PAD] = 0
    id_indexs[UNK] = 1

    for sent in sentences:
        for c in sent:
            if c not in char_number_dict:
                char_number_dict[c] = 0
            char_number_dict[c] += 1

    for c, n in char_number_dict.items():
        if n >= max_number:
            id_indexs[c] = len(id_indexs)

    return char_number_dict, id_indexs


# 返回实体列表 ([单词下标位置], 实体类型)
def get_entity(decode, text = None):
    starting = False
    p_ans = []
    for i, label in enumerate(decode):
        if label > 0:
            if label % 2 == 1:  # 1： 表示开始
                starting = True
                p_ans.append(([i], labels_text_mp[label // 2]))
            elif starting:  # 1： 表示中间
                p_ans[-1][0].append(i)
        else:
            # 0： 表示不是实体
            starting = False
    return p_ans


# 处理数据，识别每条语句中的实体信息
class Feature(object):
    def __init__(self, sent, label, id_indexs = None):
        # 文本原句子
        self.or_text = sent
        # 句子长度，最长Max_len
        self.seq_length = len(sent) if len(sent) < Max_Len else Max_Len
        # 句子每个词对应的标记
        self.labels = [LABEL_MAP[c] for c in label][:Max_Len] + [0] * (Max_Len - len(label))
        # 句子每个词对应的字典序号
        self.token_ids = self.tokenizer(sent, id_indexs)[:Max_Len] + [0] * (Max_Len - len(sent))
        # ([单词下标位置], 实体类型)
        self.entity = get_entity(self.labels, self.or_text)

    def tokenizer(self, sent, id_indexs):
        token_ids = []
        for c in sent:
            if c in id_indexs.keys():
                token_ids.append(id_indexs[c])
            else:
                token_ids.append(id_indexs[UNK])
        return token_ids


# 自定义数据生成器
class GetDatasetGenerator:
    def __init__(self, data, id_indexs = None):
        self.features = [Feature(data[0][i], data[1][i], id_indexs) for i in range(len(data[0]))]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        token_ids = feature.token_ids
        labels = feature.labels
        text = feature.or_text + [""] * (Max_Len - len(feature.or_text))
        # print("+++++")
        # print((token_ids, feature.seq_length, labels, text))
        # print("+++++")
        return (token_ids, feature.seq_length, labels, text)
        # return (token_ids, feature.seq_length, labels)


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

Max_Len = 113
batch_size = 16
UNK = "<UNK>"
PAD = "<PAD>"
NUM = "<NUM>"

# BIOES标注模式： 一般一共分为四大类：PER（人名），LOC（位置[地名]），ORG（组织）以及MISC(杂项)，而且B表示开始，I表示中间，O表示不是实体。
Entity = ['PER', 'LOC', 'ORG', 'MISC']
labels_text_mp = {k: v for k, v in enumerate(Entity)}
LABEL_MAP = {'O': 0}  # 非实体
for i, e in enumerate(Entity):
    LABEL_MAP[f'B-{e}'] = 2 * (i + 1) - 1  # 实体首字
    LABEL_MAP[f'I-{e}'] = 2 * (i + 1)  # 实体非首字

if __name__ == '__main__':

    train = read_data('../../conll2003/train.txt')
    test = read_data('../../conll2003/test.txt')
    char_number_dict, id_indexs = get_dict(train[0])

    # 预测：test
    print("------------")
    test_dataset_generator = GetDatasetGenerator(test, id_indexs)
    dataset_test = ds.GeneratorDataset(test_dataset_generator, ["data", "length", "label", "text"], shuffle=False)
    # dataset_test = ds.GeneratorDataset(test_dataset_generator, ["data", "length", "label"], shuffle=False)
    print(dataset_test.get_dataset_size())

    print("------------")
    dataset_test = dataset_test.batch(batch_size=batch_size, drop_remainder=True)
    print(dataset_test.get_dataset_size())

    for batch, (token_ids, seq_length, labels, text) in enumerate(dataset_test.create_tuple_iterator()):
        print((batch, (token_ids, seq_length, labels, text)))
        print("labels: {}".format(labels))
        print("text: {}".format(text))
        break

    for item in dataset_test:
        print(item[0].shape)
        break
