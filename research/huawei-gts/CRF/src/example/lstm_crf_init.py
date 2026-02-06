import sys
from pathlib import Path

sys.path.append(f'{Path.cwd()}')
print(f'======CurrentPath: {Path.cwd()}')
import mindspore as ms

# 设置mindspore的执行目标，可以使Ascend、CPU、GPU，mode建议位图模式。注意，ms需要放到import的首行，避免context设置不生效
ms.set_context(device_target="CPU", mode=ms.GRAPH_MODE)
from model.lstm_crf_model import BiLSTM_CRF
from utils.config import config

if __name__ == '__main__':
    # 需要使用的实体索引，可以根据需要使用BIO或者BIOES作为标注模式
    tag_to_idx = {"B": 0, "I": 1, "O": 2}

    len_id_index = 1024

    # 初始化模型
    model = BiLSTM_CRF(vocab_size=len_id_index, embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
                       num_tags=len(tag_to_idx))
    print(model)
