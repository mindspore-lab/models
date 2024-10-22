# -*- encoding: utf-8 -*-
# here put the import lib
import mindspore
import mindspore.nn as nn


class PointWiseFeedForward(nn.Cell):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = mindspore.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = mindspore.nn.Dropout(p=dropout_rate)
        self.relu = mindspore.nn.ReLU()
        self.conv2 = mindspore.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = mindspore.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class BPRLoss(nn.Cell):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

    
    def forward(self, pos_logit, neg_logit):

        loss = - mindspore.log(mindspore.sigmoid(pos_logit - neg_logit))

        return loss




class BertClassifier(nn.Cell):

    def __init__(self, user_num, item_num, device, args):
        
        super(BertClassifier, self).__init__()

        self.dev = device
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.embedding_dim = args.hidden_size
        self.guide_type = args.guide_type

        self.pos_emb = mindspore.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = mindspore.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = mindspore.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = mindspore.nn.ModuleList()
        self.forward_layernorms = mindspore.nn.ModuleList()
        self.forward_layers = mindspore.nn.ModuleList()

        self.last_layernorm = mindspore.nn.LayerNorm(args.hidden_size, eps=1e-8)

        for _ in range(args.trm_num):
            new_attn_layernorm = mindspore.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  mindspore.nn.MultiheadAttention(args.hidden_size,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = mindspore.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_size, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        if (self.guide_type == 'item') | (self.guide_type == 'seq'):
            self.loss_func = nn.BCEWithLogitsLoss(reduce=False)

        elif self.guide_type == 'cond':
            self.loss_func = nn.MSELoss()

        elif self.guide_type == 'bpr':
            self.loss_func = BPRLoss()

        else:
            raise ValueError



    def log2feats(self, seqs, positions):

        seqs *= self.embedding_dim ** 0.5  # QKV/sqrt(D)
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        timeline_mask = mindspore.zeros((seqs.shape[0], seqs.shape[1]))
        timeline_mask = timeline_mask.bool().to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        for i in range(len(self.attention_layers)):
            seqs = mindspore.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,)
                                            #attn_mask=mask)
                                            #key_padding_mask=timeline_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = mindspore.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats


    def forward(self, log_seqs, guide, mask_emb): # for training        

        log_seqs = mindspore.cat([log_seqs, mask_emb], dim=1)
        positions = mindspore.arange(1, log_seqs.shape[1]+1, device=self.dev).unsqueeze(0)
        positions = positions.repeat(log_seqs.shape[0], 1)
        log_feats = self.log2feats(log_seqs, positions) # (bs, max_len, hidden_size)
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        if self.guide_type == 'item':
            pos_logits = mindspore.mul(final_feat, guide).sum(dim=-1) # (bs, 1)
            #neg_logits = torch.mul(final_feat, neg_emb).sum(dim=-1) # (bs, 1)

            pos_labels = mindspore.ones(pos_logits.shape, device=self.dev)
            pos_loss = self.loss_func(pos_logits, pos_labels)
            loss = pos_loss
        
        elif self.guide_type == 'cond':
            loss = self.loss_func(final_feat, guide)

        elif self.guide_type == 'seq':
            mask_guide = mindspore.sum(guide > 0, dim=-1) > 0   # guide: (bs, seq_len, hidden_size), mask_guide: (bs, seq_len)
            final_feat = final_feat.unsqueeze(1).repeat(1, guide.shape[1], 1)    # (bs, hidden_size) --> (bs, seq_len, hidden_size)
            pos_logits = mindspore.mul(final_feat, guide).sum(dim=-1) # (bs, seq_len)
            pos_labels = mindspore.ones(pos_logits.shape, device=self.dev)
            pos_loss = self.loss_func(pos_logits, pos_labels) * mask_guide
            loss = mindspore.sum(pos_loss, dim=-1) / mindspore.sum(mask_guide, dim=-1)
        
        elif self.guide_type == 'bpr':
            
            pos, neg = guide
            mask_guide = mindspore.sum(pos > 0, dim=-1) > 0
            final_feat = final_feat.unsqueeze(1).repeat(1, pos.shape[1], 1)    # (bs, hidden_size) --> (bs, seq_len, hidden_size)
            pos_logits = mindspore.mul(final_feat, pos).sum(dim=-1) * mask_guide # (bs, seq_len)
            neg_logits = mindspore.mul(final_feat, neg).sum(dim=-1) * mask_guide
            loss = mindspore.mean(self.loss_func(pos_logits, neg_logits), dim=-1) / mindspore.sum(mask_guide, dim=-1)

        return loss # loss







