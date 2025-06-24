# from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM
import os
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Normal
# from mindnlp.transformers.models import BertConfig, BertForMaskedLM
# from mindformers import  BertConfig, BertForPreTraining
# from mindnlp.transformers import AutoModel
from mindspore.common.initializer import TruncatedNormal
from mindspore import context
import numpy as np
from mindspore import Tensor, Parameter
from mindspore.ops.functional import concat
from mindspore import load_checkpoint, dtype
import json
from mindspore import dtype as mstype

# from mindnlp.transformers import BertForMaskedLM
# from mindnlp.transformers import BertTokenizer
from mindformers import BertTokenizer
from mindformers import BertConfig
from mindformers import BertModel, BertTokenizer
from mindspore import nn, ops
from mindformers import BertForPreTraining, BertTokenizer
from mindformers.models.bert.bert_config import BertConfig

class BertForMaskedLM(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = nn.Dense(config.hidden_size, config.vocab_size)  # MLM输出层
    
    def construct(self, input_ids, token_type_ids=None, input_mask=None, labels=None):
        # 1. 处理可选参数
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids, dtype=ms.int32)
        if input_mask is None:
            input_mask = ops.ones_like(input_ids, dtype=ms.int32)
        
        # 2. 调用BERT模型
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            input_mask=input_mask
        )
        
        # 3. 获取序列输出并计算MLM logits
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        logits = self.cls(sequence_output)  # [batch_size, seq_len, vocab_size]
        
        # 4. 计算损失（如果提供labels）
        if labels is not None:
            loss = ops.cross_entropy(
                logits.view(-1, logits.shape[-1]),  # [batch*seq_len, vocab_size]
                labels.view(-1),                    # [batch*seq_len]
                ignore_index=-100                   # 忽略padding位置
            )
            return loss
        
        return logits  # [batch_size, seq_len, vocab_size]


def create_model(args):
    # if '11b' in args.model_name:
    #     from ..megatron_11b.megatron_wrapper import load_megatron_lm
    #     print("Warning: loading MegatronLM (11B) in fp16 requires about 28G GPU memory, and may need 3-5 minutes to load.")
    #     return load_megatron_lm(args)
    # MODEL_CLASS, _ = get_model_and_tokenizer_class(args)#获取模型类型，AutoModelForMaskedLM是一个MLM模型，可以预测被[MASK]掉位置的概率分布，[MASK]是从bert-base-cased的词汇表中选择
    # model = MODEL_CLASS.from_pretrained(r'D:\KGC\project\yf\add_tokens_plms\bert-base-cased')
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
    tokenizer = BertTokenizer.from_pretrained('bert_base_uncased')
    # 加载 BERT-MLM 模型
    model = BertForPreTraining.from_pretrained('bert_base_uncased')
    
    # 添加新词到分词器
    num_added_toks = tokenizer.add_tokens(all_entities)
    new_num_tokens = len(tokenizer)
    # # 调整模型嵌入层大小
    original_embedding = model.bert.word_embedding.embedding_table

    new_embeddings = nn.Embedding(new_num_tokens, original_embedding.shape[1])

    new_embeddings_shape = (new_num_tokens - original_embedding.shape[0], original_embedding.shape[1])
    new_embeddings_shape = np.ndarray(shape=new_embeddings_shape, dtype=np.float32)
    TruncatedNormal(0.02)(new_embeddings_shape)
    new_embeddings_shape = Parameter(Tensor(new_embeddings_shape))

    concatenated_embeddings = concat((original_embedding, new_embeddings_shape), axis=0)
    concatenated_embeddings = Parameter(Tensor(concatenated_embeddings))

    new_embeddings.embedding_table.set_data(concatenated_embeddings)
    # new_embeddings= original_embedding
    model.bert.word_embedding.embedding_table = new_embeddings.embedding_table
    model.bert.word_embedding.vocab_size = new_num_tokens
    

    # model.resize_token_embeddings(len(tokenizer))
    # # print(model.bert.embeddings.word_embeddings.embedding_size) 768
    # # print(model.bert.embeddings.word_embeddings.vocab_size) 103316
    print('----------------模型加载完成！----------------')
    def load_model_with_extended_vocab(all_entities):
        # 1. 基础模型加载
        model_path = "PLM_MODELS/bert_base_uncased"
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        
        # 2. 检查LayerNorm初始化
        for name, param in model.parameters_and_names():
            if "LayerNorm" in name and param is None:
                shape = model.get_parameter(name.replace(".weight", "").replace(".bias", "")).shape
                if "weight" in name:
                    new_param = Parameter(ops.ones(shape, ms.float32))
                else:
                    new_param = Parameter(ops.zeros(shape, ms.float32))
                model.insert_param_to_cell(name, new_param)

        # 3. 扩展词表
        num_added_toks = tokenizer.add_tokens(all_entities)
        if num_added_toks > 0:
            old_emb = model.bert.embeddings.word_embeddings
            old_vocab_size = old_emb.vocab_size
            
            # 创建新嵌入层
            new_emb = nn.Embedding(
                len(tokenizer),
                old_emb.embedding_size,
                weight=TruncatedNormal(0.02)  # 自动初始化
            )
            
            # 复制旧权重
            old_weight = old_emb.weight.clone()
            new_emb.weight[:old_vocab_size] = old_weight
            
            # 初始化新词向量（使用旧词向量均值）
            mean_vec = ops.mean(old_weight, axis=0)
            new_emb.weight[old_vocab_size:] = mean_vec
            
            # 替换嵌入层
            model.bert.embeddings.word_embeddings = new_emb
            model.resize_token_embeddings(len(tokenizer))
        
        # 4. 验证模型完整性
        for name, param in model.parameters_and_names():
            if param is None:
                raise ValueError(f"Parameter {name} is None after initialization")
    
        print('----------------模型加载完成！----------------')
        return model, tokenizer

    def load_mlm_model_with_extended_vocab(all_entities):
        # 1. 初始化
        model_path = "PLM_MODELS/bert_base_uncased"
        config = BertConfig.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # 2. 扩展词表
        num_added = tokenizer.add_tokens(all_entities)
        if num_added > 0:
            config.vocab_size += num_added  # 必须更新config
        
        # 3. 创建模型
        model = BertForMaskedLM(config)
        
        # 4. 处理新词向量
        if num_added > 0:
            old_emb = model.bert.word_embedding
            new_emb = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                embedding_table=ops.zeros((config.vocab_size, config.hidden_size), dtype=ms.float32)
            )
            
            # 复制旧权重
            new_emb.embedding_table[:old_emb.embedding_table.shape[0]] = old_emb.embedding_table
            
            # 初始化新词（使用旧词均值）
            mean_vec = ops.mean(old_emb.embedding_table, axis=0)
            new_emb.embedding_table[config.vocab_size - num_added:] = mean_vec
            
            # 更新模型
            model.bert.word_embedding = new_emb
            model.cls.weight = new_emb.embedding_table  # 绑定权重
            print('------------模型创建成功！------------')
        
        return model, tokenizer
    
    # model, tokenizer = load_model_with_extended_vocab(all_entities)

    if not args.use_lm_finetune:
        if 'megatron' in args.model_name:
            raise NotImplementedError("MegatronLM 11B is not for fine-tuning.")
        # model = model.half()#将模型的权重变成半精度浮点数，可以减少内存的使用
    return model, tokenizer


# def get_model_and_tokenizer_class(args):
#     if 'gpt' in args.model_name:
#         return GPT2LMHeadModel, AutoTokenizer
#     elif 'bert' in args.model_name:
#         return AutoModelForMaskedLM, AutoTokenizer
#     elif 'megatron' in args.model_name:
#         return None, AutoTokenizer
#     else:
#         raise NotImplementedError("This model type ``{}'' is not implemented.".format(args.model_name))


def get_embedding_layer(args, model):
    if 'roberta' in args.model_name:
        embeddings = model.bert.word_embeddings
    elif 'bert' in args.model_name:
        embeddings = model.bert.word_embedding.embedding_table
    elif 'gpt' in args.model_name:
        embeddings = model.bert.embeddings.word_embeddings
    elif 'megatron' in args.model_name:
        embeddings = model.bert.word_embeddings
    else:
        raise NotImplementedError()
    return embeddings
