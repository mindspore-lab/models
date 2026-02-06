import sys
from pathlib import Path

sys.path.append(f'{Path.cwd()}')
print(f'======CurrentPath: {Path.cwd()}')
import mindspore as ms

# 设置mindspore的执行目标，可以使Ascend、CPU、GPU，mode建议位图模式。注意，ms需要放到import的首行，避免context设置不生效
ms.set_context(device_target="CPU", mode=ms.GRAPH_MODE)
from model.lstm_crf_model import CRF

import mindspore.numpy as mnp

if __name__ == '__main__':
    # 需要使用的实体索引，可以根据需要使用BIO或者BIOES作为标注模式
    tag_to_idx = {"B": 0, "I": 1, "O": 2}

    # 初始化模型，这里需要传入待使用实体映射的个数
    model = CRF(len(tag_to_idx))

    # 定义计算loss相关的输入emissions与tag
    seq_length = 3
    batch_size = 2
    emissions = mnp.randn(seq_length, batch_size, len(tag_to_idx))
    tags = mnp.array([[0, 1], [1, 2], [2, 1]])
    loss = model(emissions, tags)
    print(loss)
