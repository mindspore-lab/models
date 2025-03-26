import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(f'{Path.cwd()}')
print(f'======CurrentPath: {Path.cwd()}')
import mindspore_lite as mslite
import mindspore.dataset as ds

from model.lstm_crf_model import CRF
from utils.dataset import read_data, GetDatasetGenerator, get_dict, COLUMN_NAME, get_entity
from utils.metrics import get_metric
from utils.config import config

if __name__ == '__main__':
    # Step1： 定义初始化参数
    batch_size = config.batch_size

    # Step2: 定义mslite运行参数
    context = mslite.Context()
    context.target = ["Ascend"]
    context.ascend.device_id = 0
    context.cpu.thread_num = 1
    context.cpu.thread_affinity_mode = 2

    # Step3: 初始化模型
    file_name = ''
    model = mslite.Model()
    model.build_from_file(file_name, mslite.ModelType.MINDIR, context)
    print(model)

    # Step4: 读取数据集
    train_dataset = read_data('../../conll2003/train.txt')
    test_dataset = read_data('../../conll2003/test.txt')
    char_number, id_indexs = get_dict(train_dataset[0])

    test_dataset_generator = GetDatasetGenerator(test_dataset, id_indexs)
    test_dataset_ds = ds.GeneratorDataset(test_dataset_generator, COLUMN_NAME, shuffle=False)
    test_dataset_batch = test_dataset_ds.batch(batch_size, drop_remainder=True)

    # Step5: 进行预测
    decodes = []
    with tqdm(total=test_dataset_batch.get_dataset_size()) as t:
        for batch, (token_ids, seq_length, labels) in enumerate(test_dataset_batch.create_tuple_iterator()):
            inputs = model.get_inputs()
            inputs[0].set_data_from_numpy(
                token_ids.asnumpy().astype(dtype=np.int32))
            inputs[1].set_data_from_numpy(
                seq_length.asnumpy().astype(dtype=np.int32))
            outputs = model.predict(inputs)
            score, history = outputs[0], outputs[1:]
            score = score.get_data_to_numpy()
            history = list(map(lambda x: x.get_data_to_numpy(), history))
            best_tags = CRF.post_decode(score, history, seq_length.asnumpy())
            decode = [[y for y in x] for x in best_tags]
            decodes.extend(list(decode))

    pred = [get_entity(x) for x in decodes]
    get_metric(pred, test_dataset_ds)
