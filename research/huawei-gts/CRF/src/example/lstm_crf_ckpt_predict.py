import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(f'{Path.cwd()}')
print(f'======CurrentPath: {Path.cwd()}')
import mindspore as ms
import mindspore.dataset as ds
from mindspore.train.serialization import load_checkpoint, load_param_into_net

# 设置mindspore的执行目标，可以使Ascend、CPU、GPU，mode建议位图模式。注意，ms需要放到import的首行，避免context设置不生效
ms.set_context(device_target="CPU", mode=ms.GRAPH_MODE)
from model.lstm_crf_model import BiLSTM_CRF, CRF
from utils.dataset import read_data, GetDatasetGenerator, get_dict, COLUMN_NAME, get_entity
from utils.metrics import get_metric
from utils.config import config

if __name__ == '__main__':
    # Step1： 定义初始化参数
    batch_size = config.batch_size

    # BIOES标注模式： 一般一共分为四大类：PER（人名），LOC（位置[地名]），ORG（组织）以及MISC(杂项)，而且B表示开始，I表示中间，O表示不是实体。
    Entity = ['PER', 'LOC', 'ORG', 'MISC']

    # Step2: 加载ckpt，传入文件路径与名称
    file_name = 'lstm_crf.ckpt'
    param_dict = load_checkpoint(file_name)

    # Step3: 获取模型初始化参数
    embedding_shape = param_dict.get('embedding.embedding_table').shape

    # Step4: 初始化模型
    model = BiLSTM_CRF(vocab_size=embedding_shape[0], embedding_dim=embedding_shape[1], hidden_dim=embedding_shape[1],
                       num_tags=len(Entity) * 2 + 1)

    # Step5: 将ckpt导入model
    load_param_into_net(model, param_dict)
    print(model)

    # Step6: 读取数据集
    train_dataset = read_data('../../conll2003/train.txt')
    test_dataset = read_data('../../conll2003/test.txt')
    char_number, id_indexs = get_dict(train_dataset[0])

    test_dataset_generator = GetDatasetGenerator(test_dataset, id_indexs)
    test_dataset_ds = ds.GeneratorDataset(test_dataset_generator, COLUMN_NAME, shuffle=False)
    test_dataset_batch = test_dataset_ds.batch(batch_size, drop_remainder=True)

    # Step7: 进行预测
    decodes = []
    model.set_train(False)
    with tqdm(total=test_dataset_batch.get_dataset_size()) as t:
        for batch, (token_ids, seq_len, labels) in enumerate(test_dataset_batch.create_tuple_iterator()):
            score, history = model(token_ids, seq_len)
            best_tag = CRF.post_decode(score, history, seq_len)
            decode = [[y for y in x] for x in best_tag]
            decodes.extend(list(decode))
            t.update(1)

    pred = [get_entity(x) for x in decodes]
    get_metric(pred, test_dataset_generator)
