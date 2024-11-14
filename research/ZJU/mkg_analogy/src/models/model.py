import numpy as np
from functools import partial
from typing import Callable, Iterable, List
import mindspore
from mindspore import nn, ops
from mindspore.experimental.optim.lr_scheduler import LinearLR


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def decode(output_ids, tokenizer):
    return lmap(str.strip, tokenizer.batch_decode(output_ids,
                                                  skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=True))

class TransformerLitModel(nn.Cell):
    def __init__(self, model, args, tokenizer, data_config={}):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = model
        self.args = args

        self.best_acc = 0
        self.first = True
        self.tokenizer = tokenizer
        self.__dict__.update(data_config)   # update config

        # resize the word embedding layer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.alpha = args.alpha

    def _init_relation_word(self):
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens':
                                                              ["[R]"]})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.decode = partial(decode, tokenizer=self.tokenizer)
        word_embeddings = self.model.get_input_embeddings()
        continous_label_word = self.analogy_relation_ids

        rel_word = self.tokenizer("[R]").tolist()
        for i, idx in enumerate(rel_word):
            word_embeddings.embedding_table[rel_word[i]] = \
                ops.mean(word_embeddings.embedding_table[continous_label_word], axis=0)

    def construct(self, input_ids, attention_mask, label, token_type_ids, pixel_values, pre_type=None,
                  sep_idx=None, q_head_idx=None, a_head_idx=None, rel_idx=None):
        model_output = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                sep_idx=sep_idx,
                                pixel_values=pixel_values,
                                return_dict=True)
        logits = model_output[0].logits
        bs = input_ids.shape[0]

        if self.args.pretrain:
            mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero()[:, 1]
            assert mask_idx.shape[0] == bs, "only one mask in sequence!"
            mask_logits = logits[ops.arange(bs), mask_idx]    # bsz, 1, vocab

            entity_loss, relation_loss = 0, 0
            entity_mask = (pre_type != 2).nonzero()[0]
            if len(entity_mask) > 0:
                entity_logits = mask_logits[entity_mask, self.entity_id_st:self.entity_id_ed]
                entity_label = label[entity_mask]
                entity_loss = self.loss_fn(entity_logits, entity_label)

            relation_mask = (pre_type == 2).nonzero()[0]
            if len(relation_mask) > 0:
                relation_logits = mask_logits[relation_mask, self.relation_id_st:self.relation_id_ed]
                relation_label = label[relation_mask]
                relation_loss = self.loss_fn(relation_logits, relation_label)

            loss = entity_loss + relation_loss

        else:
            # tail prediction
            mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero()[:, 1]    # bsz
            mask_logits = logits[ops.arange(bs), mask_idx][:, self.analogy_entity_ids]    # bsz, 1, entity

            """relaxation loss: close relation and far entity"""
            trans_hidden_states = model_output[1]   # bsz, len, hidden_size
            rel_hidden_state = trans_hidden_states[ops.arange(bs), rel_idx[ops.arange(bs), 0]] # relation between examples
            r_hidden_state = trans_hidden_states[ops.arange(bs), rel_idx[ops.arange(bs), 1]]   # relation between question and answer
            q_head_hidden_state = trans_hidden_states[ops.arange(bs), q_head_idx[ops.arange(bs)]] # relation between examples
            a_head_hidden_state = trans_hidden_states[ops.arange(bs), a_head_idx[ops.arange(bs)]]   # relation between question and answer
            sim_loss = (ops.relu(ops.cosine_similarity(q_head_hidden_state, a_head_hidden_state))
                        + 1 - ops.cosine_similarity(rel_hidden_state, r_hidden_state)).mean(0)
            loss = self.loss_fn(mask_logits, label) + self.alpha * sim_loss
        print(loss)
        return loss

    def _eval(self, input_ids, attention_mask, label, pre_type, token_type_ids, pixel_values,
                  sep_idx=None, q_head_idx=None, a_head_idx=None, rel_idx=None):
        model_output = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                sep_idx=sep_idx,
                                pixel_values=pixel_values,
                                return_dict=True)
        logits = model_output[0].logits   # bsz, len, vocab
        bs = input_ids.shape[0]

        if self.args.pretrain:
            mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero()[:, 1]
            mask_logits = logits[ops.arange(bs), mask_idx]    # bsz, 1, vocab

            entity_ranks, relation_ranks = None, None
            entity_mask = (pre_type != 2).nonzero()[0]
            if len(entity_mask) > 0:
                entity_logits = mask_logits[entity_mask, self.entity_id_st:self.entity_id_ed]   # bsz, entities
                entity_label = label[entity_mask]
                _, entity_outputs = ops.sort(entity_logits, axis=1, descending=True)           # bsz, entities   index
                _, entity_outputs = ops.sort(entity_outputs, axis=1)
                entity_ranks = entity_outputs[ops.arange(entity_mask.shape[0]), entity_label] + 1
            relation_mask = (pre_type == 2).nonzero()[0]
            if len(relation_mask) > 0:
                relation_logits = mask_logits[relation_mask, self.relation_id_st:self.relation_id_ed]
                relation_label = label[relation_mask]
                _, relation_outputs = ops.sort(relation_logits, axis=1, descending=True) # bsz, relations   index
                _, relation_outputs = ops.sort(relation_outputs, axis=1)
                relation_ranks = relation_outputs[ops.arange(relation_mask.shape[0]), relation_label] + 1

            if entity_ranks is not None and relation_ranks is not None:
                return (np.array(entity_ranks.tolist()), np.array(relation_ranks.tolist()))
            elif entity_ranks is not None:
                return (np.array(entity_ranks.tolist()), None)
            elif relation_ranks is not None:
                return (None, np.array(relation_ranks.tolist()))
            else:
                raise ValueError('entity and relation cannot be None at the same time.')

        else:
            mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero()    # bsz
            mask_logits = logits[ops.arange(bs), mask_idx][:, self.analogy_entity_ids]    # bsz, 1, entity

            _, outputs1 = ops.sort(mask_logits, axis=1, descending=True)
            _, outputs = ops.sort(outputs1, axis=1)
            entity_ranks = outputs[ops.arange(bs), label] + 1

            return (np.array(entity_ranks.tolist()), None)
