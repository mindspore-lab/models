import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer
import numpy as np
import copy

class PointWiseFeedForward(nn.Cell):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(1 - dropout_rate)
        self.relu = ops.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(1 - dropout_rate)

    def construct(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.swapaxes(-1, -2))))))

        outputs = outputs.swapaxes(-1, -2)
        outputs += inputs
        return outputs


class Bert4Rec(nn.Cell):
    def __init__(self, user_num, item_num, args):
        super(Bert4Rec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads

        self.item_emb = nn.Embedding(self.item_num + 2, args.hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(args.max_len + 100, args.hidden_size)
        self.emb_dropout = nn.Dropout(1 - args.dropout_rate)

        self.attention_layernorms = nn.CellDict()
        self.attention_layers = nn.CellDict()
        self.forward_layernorms = nn.CellDict()
        self.forward_layers = nn.CellDict()

        self.last_layernorm = nn.LayerNorm([args.hidden_size], epsilon=1e-8)

        new_attn_layernorm_1 = nn.LayerNorm([args.hidden_size], epsilon=1e-8)
        self.attention_layernorms_1 = new_attn_layernorm_1

        new_attn_layer_1 = nn.MultiheadAttention(args.hidden_size, args.num_heads, dropout=args.dropout_rate)
        self.attention_layers_1 = new_attn_layer_1

        new_fwd_layernorm_1 = nn.LayerNorm([args.hidden_size], epsilon=1e-8)
        self.forward_layernorms_1 = new_fwd_layernorm_1

        new_fwd_layer_1 = PointWiseFeedForward(args.hidden_size, dropout_rate=args.dropout_rate)
        self.forward_layers_1 = new_fwd_layer_1

        new_attn_layernorm_2 = nn.LayerNorm([args.hidden_size], epsilon=1e-8)
        self.attention_layernorms_2 = new_attn_layernorm_2

        new_attn_layer_2 = nn.MultiheadAttention(args.hidden_size, args.num_heads, dropout=args.dropout_rate)
        self.attention_layers_2 = new_attn_layer_2

        new_fwd_layernorm_2 = nn.LayerNorm([args.hidden_size], epsilon=1e-8)
        self.forward_layernorms_2 = new_fwd_layernorm_2

        new_fwd_layer_2 = PointWiseFeedForward(args.hidden_size, dropout_rate=args.dropout_rate)
        self.forward_layers_2 = new_fwd_layer_2

        self.projector = nn.Dense(args.hidden_size, args.hidden_size)
        self.loss_func = nn.BCEWithLogitsLoss()

    def log2feats(self, log_seqs, positions):
        seqs = self.item_emb(log_seqs)
        seqs *= np.sqrt(self.item_emb.embedding_size)  # QKV/sqrt(D)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == 0
        seqs *= ~timeline_mask.unsqueeze(-1)

        seqs = seqs.swapaxes(0, 1)
        Q = self.attention_layernorms_1(seqs)
        mha_outputs, _ = self.attention_layers_1(Q, seqs, seqs)
        seqs = Q + mha_outputs
        seqs = seqs.swapaxes(0, 1)

        seqs = self.forward_layernorms_1(seqs)
        seqs = self.forward_layers_1(seqs)
        seqs *= ~timeline_mask.unsqueeze(-1)

        seqs = seqs.swapaxes(0, 1)
        Q = self.attention_layernorms_2(seqs)
        mha_outputs, _ = self.attention_layers_2(Q, seqs, seqs)
        seqs = Q + mha_outputs
        seqs = seqs.swapaxes(0, 1)

        seqs = self.forward_layernorms_2(seqs)
        seqs = self.forward_layers_2(seqs)
        seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def construct(self, log_seqs, pos_seqs, neg_seqs, positions):
        log_feats = self.log2feats(log_seqs, positions)

        mask_index = (pos_seqs > 0)
        #log_feats = ops.select_index(log_feats, mask_index)
        log_feats = log_feats[mask_index]

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        #pos_embs = ops.index_select(pos_embs, mask_index)
        #neg_embs = ops.index_select(neg_embs, mask_index)
        pos_embs = pos_embs[mask_index]
        neg_embs = neg_embs[mask_index]

        pos_logits = ops.sum(log_feats * pos_embs, dim=-1)
        neg_logits = ops.sum(log_feats * neg_embs, dim=-1)

        pos_labels, neg_labels = ops.ones_like(pos_logits), ops.zeros_like(neg_logits)
        pos_loss, neg_loss = self.loss_func(pos_logits, pos_labels), self.loss_func(neg_logits, neg_labels)
        loss = pos_loss + neg_loss

        return loss

    def predict(self, log_seqs, item_indices, positions):
        log_seqs = ops.concat((log_seqs, self.mask_token * ops.ones((log_seqs.shape[0], 1)).long()), axis=-1)
        #pred_position = copy.deepcopy(positions[:, -1]) + ops.ones_like(positions[:, -1])
        #positions = ops.cat((positions, pred_position.unsqueeze(1)), axis=-1)
        log_feats = self.log2feats(log_seqs[:, 1:], positions)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(item_indices)

        logits = ops.matmul(item_embs, final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
