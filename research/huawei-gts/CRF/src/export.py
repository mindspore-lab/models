import os
import random
from datetime import datetime

import mindspore as ms

from utils.config import config

ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from tqdm import tqdm

from model.lstm_crf_model import BiLSTM_CRF

from utils.dataset import read_data, get_dict, GetDatasetGenerator, Entity, COLUMN_NAME


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
    ms.dataset.config.set_seed(seed)


if __name__ == '__main__':
    print(config)
    print(f'============# [DEVICE] {ms.get_context("device_target")}')
    print(f'============# [DEVICE_ID] {ms.get_context("device_id")}')
    print(f'============# [DEVICE_MODE] {ms.get_context("mode")}')

    seed = 42
    seed_everything(seed)

    train = read_data(config.data_path + '/train.txt')

    char_number_dict, id_indexs = get_dict(train[0])

    batch_size = config.batch_size
    dataset_generator = GetDatasetGenerator(train, id_indexs)
    dataset = ds.GeneratorDataset(dataset_generator, COLUMN_NAME, shuffle=False)
    dataset_train = dataset.batch(batch_size=batch_size)

    model = BiLSTM_CRF(vocab_size=len(id_indexs), embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
                       num_tags=len(Entity) * 2 + 1)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=config.learning_rate)
    grad_fn = ms.value_and_grad(model, None, optimizer.parameters)


    def train_step(token_ids, seq_length, labels):
        loss, grads = grad_fn(token_ids, seq_length, labels)
        optimizer(grads)
        return loss


    print('=====================# [START]训练 ==========================')
    # 训练
    tloss = []
    for epoch in range(config.num_epochs):
        model.set_train()
        with tqdm(total=dataset_train.get_dataset_size()) as t:
            for batch, (token_ids, seq_length, labels) in enumerate(dataset_train.create_tuple_iterator()):
                loss = train_step(token_ids, seq_length, labels)
                tloss.append(loss.asnumpy())
                t.set_postfix(loss=np.array(tloss).mean())
                t.update(1)

    print('=====================# [START]导出 ==========================')
    now = datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M")
    file_name = f'{config.export_prefix}-{date_time}-{config.export_suffix}-{ms.get_context("device_target")}'

    ms.export(model, ops.ones((config.batch_size, config.vocab_max_length), ms.int64),
              ops.ones(config.batch_size, ms.int64),
              file_name=file_name, file_format="MINDIR")
    print(f"export mindir success : {file_name}")

    ms.save_checkpoint(model, f"{file_name}.ckpt")
    print(f"export ckpt success : {file_name}")
