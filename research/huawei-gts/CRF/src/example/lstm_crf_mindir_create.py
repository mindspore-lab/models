import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(f'{Path.cwd()}')
print(f'======CurrentPath: {Path.cwd()}')
import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.nn as nn

# 设置mindspore的执行目标，可以使Ascend、CPU、GPU，mode建议位图模式。注意，ms需要放到import的首行，避免context设置不生效
ms.set_context(device_target="CPU", mode=ms.GRAPH_MODE)
from model.lstm_crf_model import BiLSTM_CRF
from utils.dataset import read_data, GetDatasetGenerator, get_dict, COLUMN_NAME
from utils.config import config

if __name__ == '__main__':
    # Step1： 定义初始化参数
    embedding_dim = config.embedding_dim
    hidden_dim = config.hidden_dim
    Max_Len = config.vocab_max_length
    batch_size = config.batch_size

    # BIOES标注模式： 一般一共分为四大类：PER（人名），LOC（位置[地名]），ORG（组织）以及MISC(杂项)，而且B表示开始，I表示中间，O表示不是实体。
    Entity = ['PER', 'LOC', 'ORG', 'MISC']
    labels_text_mp = {k: v for k, v in enumerate(Entity)}
    LABEL_MAP = {'O': 0}  # 非实体
    for i, e in enumerate(Entity):
        LABEL_MAP[f'B-{e}'] = 2 * (i + 1) - 1  # 实体首字
        LABEL_MAP[f'I-{e}'] = 2 * (i + 1)  # 实体非首字

    # Step2: 读取数据集
    train_dataset = read_data('../../conll2003/train.txt')
    char_number, id_indexs = get_dict(train_dataset[0])

    train_dataset_generator = GetDatasetGenerator(train_dataset, id_indexs)
    train_dataset_ds = ds.GeneratorDataset(train_dataset_generator, COLUMN_NAME, shuffle=False)
    train_dataset_batch = train_dataset_ds.batch(batch_size, drop_remainder=True)

    # Step3: 初始化模型与优化器
    model = BiLSTM_CRF(vocab_size=len(id_indexs), embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                       num_tags=len(Entity) * 2 + 1)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=config.learning_rate)
    grad_fn = ms.value_and_grad(model, None, optimizer.parameters)


    def train_step(token_ids, seq_len, labels):
        loss, grads = grad_fn(token_ids, seq_len, labels)
        optimizer(grads)
        return loss


    # Step5: 训练
    tloss = []
    for epoch in range(config.num_epochs):
        model.set_train()
        with tqdm(total=train_dataset_batch.get_dataset_size()) as t:
            for batch, (token_ids, seq_len, labels) in enumerate(train_dataset_batch.create_tuple_iterator()):
                loss = train_step(token_ids, seq_len, labels)
                tloss.append(loss.asnumpy())
                t.set_postfix(loss=np.array(tloss).mean())
                t.update(1)

    # Step6: 导出MindIR
    file_name = 'lstm_crf.mindir'
    ms.export(model, ops.ones((batch_size, Max_Len), ms.int64), ops.ones(batch_size, ms.int64), file_name=file_name,
              file_format='MINDIR')
    print(f'======Create MINDIR SUCCEEDED, file: {file_name}')
