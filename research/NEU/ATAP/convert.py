from mindspore import Tensor, save_checkpoint
from mindspore import load_checkpoint, dtype
import json
import os

from mindformers import BertTokenizer, BertConfig, BertForPreTraining
# from mindnlp.transformers import AutoModel
from mindspore.common.initializer import TruncatedNormal
from mindspore import context
import numpy as np
from mindspore import Tensor, Parameter
from mindspore.ops.functional import concat
from mindspore import load_checkpoint, dtype,  load_param_into_net
from mindspore import dtype as mstype

# # 重新加载并保存checkpoint
# param_dict = {}
# original_ckpt = load_checkpoint("add_tokens_plms/bert-base-uncased/bert_base_uncased.ckpt")
# for name, param in original_ckpt.items():
#     param_dict[name] = Tensor(param.asnumpy(), dtype=dtype.float32)  # 确保数据类型
    
# save_checkpoint(param_dict, "add_tokens_plms/bert-base-uncased/converted_bert_base_uncased.ckpt")

with open(os.path.join('PLM_MODELS/add-bert-base-uncased', 'config.json'), "r") as f:
    config_dict = json.load(f)

config = BertConfig(**config_dict)

# 初始化BertForPreTraining模型
model = BertForPreTraining(config)

# 加载 checkpoint 文件
param_dict = load_checkpoint(os.path.join('PLM_MODELS/add-bert-base-uncased', 'mindspore_model.ckpt'), net=model)

# 将加载的权重参数导入到模型
# load_param_into_net(model, param_dict)

# 初始化Tokenizer
tokenizer = BertTokenizer.from_pretrained('PLM_MODELS/add-bert-base-uncased')

print('successed')