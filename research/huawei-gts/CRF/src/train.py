import mindspore as ms
import os
import numpy as np
import random
import mindspore.nn as nn
import mindspore.dataset as ds
from tqdm import tqdm

from utils.dataset import read_data, get_dict, GetDatasetGenerator, Entity
from model.lstm_crf_model import BiLSTM_CRF


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
    ms.dataset.config.set_seed(seed)


def debug_dataset(dataset, batch_size=16):
    dataset = dataset.batch(batch_size=batch_size)
    for data in dataset.create_dict_iterator():
        print(data["data"].shape, data["label"].shape)
        break


if __name__ == '__main__':

    seed = 42
    seed_everything(seed)

    train = read_data('../conll2003/train.txt')
    test = read_data('../conll2003/test.txt')
    dev = read_data('../conll2003/valid.txt')

    cut = 5
    train = (train[0][:cut], train[1][:cut])
    test = (test[0][:cut], test[1][:cut])
    dev = (dev[0][:cut], dev[1][:cut])
    char_number_dict, id_indexs = get_dict(train[0])

    Epoch = 2
    batch_size = 16
    dataset_generator = GetDatasetGenerator(train, id_indexs)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "length", "label", "text"], shuffle=False)
    dataset_train = dataset.batch(batch_size=batch_size)

    model = BiLSTM_CRF(vocab_size=len(id_indexs), embedding_dim=128, hidden_dim=128, num_tags=len(Entity) * 2 + 1)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)
    grad_fn = ms.value_and_grad(model, None, optimizer.parameters)


    def train_step(token_ids, seq_length, labels):
        loss, grads = grad_fn(token_ids, seq_length, labels)
        optimizer(grads)
        return loss


    print('=====================# [START]训练 ==========================')
    # 训练
    size = dataset_train.get_dataset_size()
    steps = size
    tloss = []
    for epoch in range(Epoch):
        model.set_train()
        with tqdm(total=steps) as t:
            for batch, (token_ids, seq_length, labels, text) in enumerate(dataset_train.create_tuple_iterator()):
                loss = train_step(token_ids, seq_length, labels)
                tloss.append(loss.asnumpy())
                t.set_postfix(loss=np.array(tloss).mean())
                t.update(1)
