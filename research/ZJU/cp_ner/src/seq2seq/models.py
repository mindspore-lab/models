import os
from copy import deepcopy
import mindspore
from mindspore import nn
from mindspore import ops
from .t5 import T5ForConditionalGeneration

class T5Prompt(nn.Cell):
    def __init__(self, model_name_or_path, args):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.config = self.t5.config
        self.args = args

        # model config related
        self.match_n_layer = self.config.num_layers
        self.match_n_head = self.config.num_heads
        self.n_embd = self.config.hidden_size
        self.match_n_embd = self.config.kv_size

        # prefix related
        self.prompt_len = args.prompt_len
        self.prompt_dim = args.prompt_dim
        self.prompt_inputs = mindspore.numpy.arange(start=0,
                                                    stop=self.prompt_len).long()

        self.wte = nn.Embedding(self.prompt_len, self.n_embd)
        self.control_trans = nn.SequentialCell(
            nn.Dense(self.n_embd, self.prompt_dim),
            nn.Tanh(),
            nn.Dense(self.prompt_dim,
                    self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        self.wte_enc = nn.Embedding(self.prompt_len, self.n_embd)
        self.control_trans_enc = nn.SequentialCell(
            nn.Dense(self.n_embd, self.prompt_dim),
            nn.Tanh(),
            nn.Dense(self.prompt_dim,
                    self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        self.wte_dec = nn.Embedding(self.prompt_len, self.n_embd)
        self.control_trans_dec = nn.SequentialCell(
            nn.Dense(self.n_embd, self.prompt_dim),
            nn.Tanh(),
            nn.Dense(self.prompt_dim,
                    self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        self.dropout = nn.Dropout(0.1)

        if args.source_prefix_path is not None and 'concat' in args.prefix_fusion_way:
            self.source_prompt_inputs = mindspore.numpy.arange(self.prompt_len).long()
            self.source_wte = nn.Embedding(self.prompt_len, self.n_embd)
            self.source_control_trans = nn.SequentialCell(
                nn.Dense(self.n_embd, self.prompt_dim),
                nn.Tanh(),
                nn.Dense(self.prompt_dim,
                        self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )

            self.source_wte_enc = nn.Embedding(self.prompt_len, self.n_embd)
            self.source_control_trans_enc = nn.SequentialCell(
                nn.Dense(self.n_embd, self.prompt_dim),
                nn.Tanh(),
                nn.Dense(self.prompt_dim,
                         self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )

            self.source_wte_dec = nn.Embedding(self.prompt_len, self.n_embd)
            self.source_control_trans_dec = nn.SequentialCell(
                nn.Dense(self.n_embd, self.prompt_dim),
                nn.Tanh(),
                nn.Dense(self.prompt_dim,
                        self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )

        if args.multi_source_path:
            # module
            self.prefix_project_layer = nn.SequentialCell(
                nn.Dense(self.n_embd, 500),
                nn.Tanh(),
                nn.Dense(500, self.n_embd)
            )

        if 'lstm' in args.prefix_fusion_way:
            self.lstm = nn.LSTM(input_size=self.match_n_head * self.match_n_embd,
                                hidden_size=self.match_n_head * self.match_n_embd,
                                num_layers=2,
                                batch_first=True,
                                bidirectional=True)

        if args.freeze_plm:
            for param in self.t5.trainable_params():
                if 'encoder' in param.name or 'decoder' in param.name \
                    and 'adapter' not in param.name:
                    param.requires_grad = False

        if args.freeze_prefix:
            for param in self.trainable_params():
                if 'wte' in param.name or 'control_trans' in param.name:
                    param.requires_grad = False

        self.label_word_id = []


    def get_prompt(self, bsz=None):
        input_tokens = self.prompt_inputs.unsqueeze(0).broadcast_to((bsz, 10))
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        past_key_values_dec = self.control_trans_dec(
            temp_control_dec
        )  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (self.prompt_inputs.unsqueeze(0).broadcast_to((bsz, 10)))
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0],
                "prev_value": key_val[1],
                "prev_key_padding_mask": ops.zeros((bsz, seqlen))
                    .bool()
                # bsz, prompt_len
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0],
                "prev_value": key_val_dec[1],
                "prev_key_padding_mask": ops.zeros((bsz, seqlen))
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0],
                "prev_value": key_val_enc[1],
                "prev_key_padding_mask": ops.zeros((bsz_enc, seqlen))
                    .bool(),
            }
            result.append(temp)

        return result

    def convert_prefix(self, prefix_dict):
        '''trans dict prefix to 1-D tensor prefix'''
        trans_prefix = []
        for i, prefix in enumerate(prefix_dict): # 12 layer
            # 'decoder_prompt', 'cross_attention_prompt', 'encoder_prompt'
            current_key_prefix = []
            for key in prefix.keys():

                current_key_prefix.append(ops.stack([
                            prefix[key]['prev_key'].squeeze(0).swapaxes(1, 0) \
                                    .reshape(-1,self.n_embd).mean(0),
                            prefix[key]['prev_value'].squeeze(0).swapaxes(1, 0) \
                                .reshape(-1, self.n_embd).mean(0),
                        ], axis=0).mean(0))  #  768
            trans_prefix.append(ops.stack(current_key_prefix, axis=0).mean(0))
        trans_prefix = ops.stack(trans_prefix, axis=0).mean(0)
        return trans_prefix

    def normalize_multi_source(self, label_word, tokenizer, device):
        '''This function is to normalize multi source to aggregate them.'''
        assert self.args.multi_source_path is not None

        # load multi source prefix and label word
        source_paths = self.args.multi_source_path.split(',')
        self.source_prefixes = [mindspore.load(os.path.join(path, 'prefix.mindir')) \
                                for path in source_paths]

        trans_source_prefixes = []  # n x 768
        for source_prefix in self.source_prefixes: # n source
            trans_source_prefixes.append(self.convert_prefix(source_prefix))
        self.trans_source_prefixes = ops.stack(trans_source_prefixes, axis=0)

        self.trans_target_prefix = self.convert_prefix(self.get_prompt(bsz=1)).unsqueeze(0)

        source_label_words = [mindspore.load(os.path.join(path, 'label_word.mindir')) \
                              for path in source_paths]
        for i in range(len(source_label_words)):
            source_label_words[i] = ops.stack(list(source_label_words[i].values()), axis=0).mean(0)
        self.source_label_words = ops.stack(source_label_words, axis=0).to(device)
        self.label_word_id = [tokenizer.encode(label, add_special_tokens=False) \
                              for label in label_word]

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        return_dict=True,
    ):
        bsz = input_ids.shape[0]
        past_prompt = self.get_prompt(bsz=bsz)

        if self.args.multi_source_path is not None:
            # 1. compute cosine similarity between target prefix and source prefixes
            trans_source_prefixes = self.prefix_project_layer(self.trans_source_prefixes)

            trans_target_prefix = self.convert_prefix(self.get_prompt(bsz=1))
            trans_target_prefix = self.prefix_project_layer(trans_target_prefix).unsqueeze(0)

            prefix_sim = ops.cosine_similarity(trans_target_prefix, trans_source_prefixes)   # n
            prefix_sim = prefix_sim / prefix_sim.sum()

            # 2. compute cosine similarity between target label word and source label words
            target_label_word = []
            for label_id in self.label_word_id:
                if len(label_id) > 1:
                    target_label_word.append(ops.stack([self.t5.shared.embedding_table.data[id] \
                                                        for id in label_id], axis=0).mean(0))
                else:
                    target_label_word.append(self.t5.shared.embedding_table.data[label_id[0]])
            target_label_word = ops.stack(target_label_word, axis=0).mean(0)
            label_word_sim = ops.cosine_similarity(target_label_word.unsqueeze(0),
                                                   self.source_label_words)
            label_word_sim = label_word_sim / label_word_sim.sum()

            total_sim = (prefix_sim + label_word_sim) / 2
            total_sim = prefix_sim

            # aggregate
            for i, prefix in enumerate(past_prompt):    # 24
                for key in prefix.keys():
                    for sub_key in prefix[key].keys():
                        if len(prefix[key][sub_key].shape) == 4:
                            agg_prefix = ops.zeros((1, self.match_n_head, self.prompt_len,
                                                    self.match_n_embd))
                            for j in range(len(self.source_prefixes)):
                                agg_prefix += self.source_prefixes[j][i][key][sub_key] \
                                    * total_sim[j]
                            past_prompt[i][key][sub_key] = (past_prompt[i][key][sub_key] \
                                                            + agg_prefix) / 2

        return self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            past_prompt=past_prompt,
        )

    def generate(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):
        # bsz = input_ids.shape[0]
        bsz = len(input_ids)
        past_prompt = self.get_prompt(bsz=bsz)

        if self.args.multi_source_path:
            # 1. compute cosine similarity between target prefix and source prefixes
            trans_source_prefixes = self.prefix_project_layer(self.trans_source_prefixes)

            trans_target_prefix = self.convert_prefix(self.get_prompt(bsz=1))
            trans_target_prefix = self.prefix_project_layer(trans_target_prefix)\
                                    .detach().unsqueeze(0)

            prefix_sim = ops.cosine_similarity(trans_target_prefix, trans_source_prefixes)   # n
            prefix_sim = prefix_sim / prefix_sim.sum()

            # 2. compute cosine similarity between target label word and source label words
            target_label_word = []
            for label_id in self.label_word_id:
                if len(label_id) > 1:
                    target_label_word.append(ops.stack([self.t5.shared.weight.data[id] \
                                                        for id in label_id], axis=0).mean(0))
                else:
                    target_label_word.append(self.t5.shared.weight.data[label_id[0]])
            target_label_word = ops.stack(target_label_word, axis=0).mean(0)
            label_word_sim = ops.cosine_similarity(target_label_word.unsqueeze(0),
                                                   self.source_label_words)
            label_word_sim = label_word_sim / label_word_sim.sum()

            total_sim = (prefix_sim + label_word_sim) / 2

            # aggregate
            for i, prefix in enumerate(past_prompt):    # 24
                for key in prefix.keys():
                    for sub_key in prefix[key].keys():
                        if len(prefix[key][sub_key].shape) == 4:
                            agg_prefix = ops.zeros((1, self.match_n_head, self.prompt_len,
                                                    self.match_n_embd))
                            for j in range(len(self.source_prefixes)):
                                agg_prefix += self.source_prefixes[j][i][key][sub_key] \
                                    * total_sim[j]
                            past_prompt[i][key][sub_key] = (past_prompt[i][key][sub_key] + \
                                                            agg_prefix) / 2

        generated_ids = self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids
