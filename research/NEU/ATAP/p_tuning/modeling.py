import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer
from data_utils.vocab import *
# from mindnlp.transformers import AutoTokenizer
import os
from os.path import join
from p_tuning.models import get_embedding_layer, create_model
# from data_utils.vocab import get_vocab_by_strategy
from data_utils.dataset import load_file
from p_tuning.prompt_encoder import PromptEncoder
# from mindformers import BertTokenizer
import sys
from mindspore import Tensor
import mindspore.common.dtype as mstype
import numpy as np
from mindformers import BertTokenizer

class PTuneForLAMA(nn.Cell):
    def __init__(self, args, template):
        super().__init__()
        self.args = args
        
        # Load relation templates
        self.relation_templates = dict(
            (d['relation'], d['template']) for d in load_file(join(self.args.data_dir, 'relations.jsonl')))
        
        # Load tokenizer
        tokenizer_src = 'roberta-large' if 'megatron' in self.args.model_name else self.args.model_name
        # self.tokenizer = AutoTokenizer.from_pretrained(r'add_tokens_plms/bert-large-cased', use_fast=False)
        self.tokenizer_old = BertTokenizer.from_pretrained('PLM_MODELS/bert_base_uncased')
        
        # Load pre-trained model
        if 'megatron' in self.args.model_name and self.args.use_lm_finetune:
            raise RuntimeError("Can not apply args.use_lm_finetune=True on MegatronLM 11B.")
        
        self.model, self.tokenizer= create_model(self.args)
        for param in self.model.get_parameters():
            param.requires_grad = self.args.use_lm_finetune
            
        self.embeddings = get_embedding_layer(self.args, self.model)
        self.vocab = self.tokenizer.get_vocab()
        
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))
        
        if 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
            template = (template[0], template[1], 0)
        self.template = template
        # Initialize prompt encoder
        # self.hidden_size = self.embeddings.shape[1]embedding_size
        self.hidden_size = self.embeddings.shape[1]
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
        
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, args)
        
        self.pad_op = ops.Pad(((0, 0), (0, 128)))


    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = ops.identity(queries)
        replace_mask = (queries == self.pseudo_token_id)
        queries_for_embedding = ops.masked_fill(queries_for_embedding, replace_mask, self.tokenizer.unk_token_id)

        if self.args.use_original_template:
            return queries_for_embedding
         
        replace_mask = ops.Cast()(replace_mask, ms.int32)
        if len(replace_mask.shape) == 1:
            replace_mask = replace_mask.reshape(bz, -1)
        nonzero_indices = ops.nonzero(replace_mask)
        # blocked_indices = nonzero_indices.reshape((bz, self.spell_length, 2))[:, :, 1]
        blocked_indices = nonzero_indices[:, 1].reshape(bz, -1)
        
        replace_embeds = self.prompt_encoder()
        replace_embeds = ops.Cast()(replace_embeds, ms.int32) 

        for bidx in range(bz):
            for i in range(self.spell_length):
                queries_for_embedding[bidx, blocked_indices[bidx, i]] = replace_embeds[i]
        return queries_for_embedding

    def get_query(self, head_ids, x_h, prompt_tokens, x_t=None):

        # For P-tuning
        if 'gpt' not in self.args.model_name and 'megatron' not in self.args.model_name:    
            if not isinstance(head_ids, list):  
                head_ids = head_ids.asnumpy().tolist()
            query = [self.tokenizer.cls_token_id]+ prompt_tokens * self.template[0] + head_ids + prompt_tokens * self.template[1] + [self.tokenizer.mask_token_id]

            if self.template[2] > 0:
                query += (prompt_tokens * self.template[2])
            else:
                query += (self.tokenizer.convert_tokens_to_ids(['.']))
                
            query += [self.tokenizer.sep_token_id]  # [SEP]
            return query

        else:
            raise NotImplementedError(f"The query template for {self.args.model_name} has not been defined.")

    def construct(self,  x_hs, label_ids, evaluate_type, token_ids=None, epoch=1, return_candidates=False):            
        bz = label_ids.shape[0]     
        prompt_tokens = [self.pseudo_token_id]

        queries = []
        for i in range(bz):
            query = self.get_query(token_ids[i], x_hs[i], prompt_tokens)
            if query:
                queries.append(ms.Tensor(query, dtype=ms.int32))
                
        if not queries:
            raise ValueError("No valid queries were generated. Check your input data and get_query method.")
        # max_len = max(q.shape[0] for q in queries)
        max_len = 128
        padded_queries = ops.zeros((bz, max_len), dtype=ms.int32)
        for i, q in enumerate(queries):
            valid_length = min(int(q.shape[0]), max_len)
            if valid_length > 0:
                padded_queries[i, :valid_length] = q[:valid_length]
 
        # get embedded input
        inputs_embeds = self.embed_input(padded_queries)

        attention_mask = padded_queries != self.pad_token_id
        attention_mask = ops.Cast()(attention_mask, ms.int32)

        token_type_ids = ops.zeros_like(attention_mask, dtype=ms.int32)
        
        # Find mask positions
        mask_pos = (padded_queries == 101)
        # mask_pos = (padded_queries == 101)
        nonzero_indices = ops.nonzero(mask_pos)  
        label_mask = nonzero_indices[:, 1].reshape(-1, 1)
        # label_mask = nonzero_indices.reshape((bz, 1))[:, 1:2]  # [bz, n_masks, 1]

        masked_lm_weights = ops.ones((bz, 1), dtype=ms.float32)
        ooo = ops.ones((bz, 1), dtype=ms.int32)
        label_ids = ops.reshape(label_ids, (bz, 1)) 
        label_mask = ops.reshape(label_mask, (bz, 1)) 
        
        # label_mask = label_mask.squeeze(1)
        # masked_lm_weights = masked_lm_weights.squeeze(1)
        random_tensor = ops.randint( 0, 100,(1, 128), dtype=ms.int32)
        output = self.model(
            input_ids=inputs_embeds,
            input_mask=attention_mask,
            token_type_id=token_type_ids,

            masked_lm_positions=label_mask, # 掩码词的位置
            next_sentence_labels=label_ids,
            masked_lm_weights = masked_lm_weights,  # 掩码词的权重
            masked_lm_ids = label_ids
        )
        # loss, logits = output[0], output[1]
        if evaluate_type == 'Test':
            loss, logits = output[-1], output[-2]
            for batch_size in range(bz):
                if label_ids[batch_size, 0] >= 28996:
                    logits[batch_size, :28995] = -100
                else:
                    logits[batch_size, 28995:] = -100

            pred_ids = ops.Sort(descending=True)(logits)[1]  # #按照logits的第2维度由大到小进行排序，排序的结果按照索引进行显示。（bz*28996）    

            # return label_ids,pred_ids,1,1,1
            hit1, hit3, hit10, mrr = 0, 0, 0, 0
            for i in range(bz):
                # pred_seq = pred_ids[i].asnumpy().tolist() # (108861)
                preds = pred_ids[i].asnumpy().flatten()
                label = int(label_ids[i, 0])
                ranks = np.where(preds == label)[0]
                if len(ranks) > 0:
                    rank = ranks[0] + 1  # 转换为1-based
                    if rank <= 1: hit1 += 1
                    if rank <= 3: hit3 += 1
                    if rank <= 10: hit10 += 1
                    if evaluate_type == 'Test':
                        mrr += 1.0 / rank
            return loss, hit1, hit3, hit10, mrr
        else:
            loss = output[0]
            return loss, 1
        # return output, logits, 1, 1, 1
        