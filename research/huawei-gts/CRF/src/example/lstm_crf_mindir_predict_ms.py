import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(f'{Path.cwd()}')
print(f'======CurrentPath: {Path.cwd()}')
import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn

# 设置mindspore的执行目标，可以使Ascend、CPU、GPU，mode建议位图模式。注意，ms需要放到import的首行，避免context设置不生效
ms.set_context(device_target="CPU", mode=ms.GRAPH_MODE)
from model.lstm_crf_model import CRF
from utils.dataset import read_data, GetDatasetGenerator, get_dict, COLUMN_NAME, get_entity
from utils.metrics import get_metric
from utils.config import config

if __name__ == '__main__':
    # Step1： 定义初始化参数
    batch_size = config.batch_size

    # Step2: 通过mindir加载模型
    file_name = 'lstm_crf.mindir'
    graph = ms.load(file_name)
    model = nn.GraphCell(graph)
    print(model)

    # Step3: 读取数据集
    train_dataset = read_data('../../conll2003/train.txt')
    test_dataset = read_data('../../conll2003/test.txt')
    char_number, id_indexs = get_dict(train_dataset[0])

    test_dataset_generator = GetDatasetGenerator(test_dataset, id_indexs)
    test_dataset_ds = ds.GeneratorDataset(test_dataset_generator, COLUMN_NAME, shuffle=False)
    test_dataset_batch = test_dataset_ds.batch(batch_size, drop_remainder=True)

    # Step4: 进行预测
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
