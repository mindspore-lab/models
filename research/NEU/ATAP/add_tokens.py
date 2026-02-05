# from transformers import BertForMaskedLM,BertTokenizer

# all_entities = []
# with open('ATAP_code/atap/data/ConceptNet/train.txt','r',encoding='utf-8') as f:
#     triples = f.readlines()
    
#     for triple in triples:
#         relation,head,tail = triple.strip().split('\t')
#         if head not in all_entities:
#             all_entities.append(head)
#         if tail not in all_entities:
#             all_entities.append(tail)
# with open('ATAP_code/atap/data/ConceptNet/test.txt','r',encoding='utf-8') as f:
#     triples = f.readlines()
    
#     for triple in triples:
#         relation,head,tail = triple.strip().split('\t')
#         if head not in all_entities:
#             all_entities.append(head)
#         if tail not in all_entities:
#             all_entities.append(tail)
# with open('ATAP_code/atap/data/ConceptNet/valid.txt','r',encoding='utf-8') as f:
#     triples = f.readlines()
#     for triple in triples:
#         relation,head,tail = triple.strip().split('\t')
#         if head not in all_entities:
#             all_entities.append(head)
#         if tail not in all_entities:
#             all_entities.append(tail)


# model = r"PLM_MODELS/bert_base_uncased"
# tokenizer = BertTokenizer.from_pretrained(model)
# model = BertForMaskedLM.from_pretrained(model)
# num_added_toks = tokenizer.add_tokens(all_entities)
# model.resize_token_embeddings(len(tokenizer))
# tokenizer.save_pretrained(r"PLM_MODELS/add-bert-base-uncased")
# model.save_pretrained(r"PLM_MODELS/add-bert-base-uncased")

import os
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Normal
# from mindnlp.transformers.models import BertConfig, BertForMaskedLM
from mindformers import BertTokenizer, BertConfig, BertForPreTraining
# from mindnlp.transformers import AutoModel
from mindspore.common.initializer import TruncatedNormal
from mindspore import context
import numpy as np
from mindspore import Tensor, Parameter
from mindspore.ops.functional import concat
from mindspore import load_checkpoint, dtype
import json
from mindspore import dtype as mstype

# 设置设备环境（GPU/CPU）
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

all_entities = []
# 读取实体数据
def load_entities(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        triples = f.readlines()
        for triple in triples:
            relation, head, tail = triple.strip().split('\t')
            if head not in all_entities:
                all_entities.append(head)
            if tail not in all_entities:
                all_entities.append(tail)

# 加载所有实体
load_entities('ATAP_code/atap/data/ConceptNet/train.txt')
load_entities('ATAP_code/atap/data/ConceptNet/test.txt')
load_entities('ATAP_code/atap/data/ConceptNet/valid.txt')

# 初始化模型和分词器
model_path = "PLM_MODELS/bert_base_uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForPreTraining.from_pretrained(model_path)

# 添加新词到分词器
num_added_toks = tokenizer.add_tokens(all_entities)
new_num_tokens = len(tokenizer)
# 调整模型嵌入层大小
original_embedding = model.bert.word_embedding.embedding_table
new_embedding = nn.Embedding(new_num_tokens, original_embedding.shape[1])

new_embeddings_shape = (new_num_tokens - original_embedding.shape[0], original_embedding.shape[1])
new_embeddings_shape = np.ndarray(shape=new_embeddings_shape, dtype=np.float32)
TruncatedNormal(0.02)(new_embeddings_shape)
new_embeddings_shape = Parameter(Tensor(new_embeddings_shape))

concatenated_embeddings = concat((original_embedding, new_embeddings_shape), axis=0)
concatenated_embeddings = Parameter(Tensor(concatenated_embeddings))

new_embedding.embedding_table.set_data(concatenated_embeddings)

# new_embedding.embedding_table[:original_embedding.shape[0], :] = original_embedding
model.bert.word_embedding.embedding_table = new_embedding.embedding_table
# import sys
# sys.exit()
# 保存模型和分词器

assert len(tokenizer) > 0, "分词器词汇表为空！"
print(f"最终词汇量: {len(tokenizer)}")

output_dir = "ATAP_code/atap/new/add-bert-base-uncased"
os.makedirs(output_dir, exist_ok=True)
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

# 手动保存config（关键添加）
config = BertConfig(
    vocab_size=len(tokenizer),  # 更新后的词汇量
    hidden_size=768,
    max_position_embeddings=512
    # 其他参数保持与原始config一致
)
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(config.to_dict(), f, indent=2)

print('successed!!')