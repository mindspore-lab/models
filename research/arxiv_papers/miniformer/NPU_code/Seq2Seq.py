

import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import One, Normal
import numpy as np

class Seq2SeqSum(nn.Cell):
    def __init__(self, src_vocab_size,tgt_vocab_size, emb_dim,
                 n_hidden, n_layer=1, bi_enc=True, dropout=0.0):
        super(Seq2SeqSum, self).__init__()

        self.n_layer = n_layer
        self.bi_enc = bi_enc  # whether encoder is bidirectional
        self.embedding = nn.Embedding(src_vocab_size, emb_dim, padding_idx=0)

        self.encoder = nn.LSTM(
            emb_dim, n_hidden, n_layer,
            bidirectional  = bi_enc,
            dropout  = 0.0 if n_layer==1 else dropout
        )

        states_size0 = n_layer * (2 if bi_enc else 1)
        self.enc_init_h=mindspore.ops.uniform((states_size0, n_hidden), mindspore.Tensor(-1e-2), mindspore.Tensor(1e-2))
        self.enc_init_c=mindspore.ops.uniform((states_size0, n_hidden), mindspore.Tensor(-1e-2), mindspore.Tensor(1e-2))

        self.enc_out_dim = n_hidden * (2 if bi_enc else 1)
        self._dec_h = nn.Dense(self.enc_out_dim, n_hidden, has_bias=False)
        self._dec_c = nn.Dense(self.enc_out_dim, n_hidden, has_bias=False)

        self.decoder = AttnDecoder(
            self.embedding, n_hidden, tgt_vocab_size,
            self.enc_out_dim, n_layer,
            dropout=dropout
            )

    def construct(self, src, src_lengths, tgt):

        enc_outs, init_dec_states = self.encode(src, src_lengths)
        attn_mask = len_mask(src_lengths)
        logit = self.decoder(tgt, init_dec_states, enc_outs, attn_mask)
        return logit

    def encode(self, src, src_lengths):

        size = (
            self.enc_init_c.shape[0],
            len(src_lengths),
            self.enc_init_c.shape[1]
        )

        init_hidden = (
            self.enc_init_h.unsqueeze(1).broadcast_to(size).astype(mindspore.float16),
            self.enc_init_c.unsqueeze(1).broadcast_to(size).astype(mindspore.float16)
        )

        embed = self.embedding(src.swapaxes(0, 1))
        enc_out, hidden = self.encoder(embed, init_hidden)
        outputs=enc_out

        if self.bi_enc:
            h, c = hidden
            h= mindspore.ops.cat(mindspore.ops.chunk(h,2,axis=0), axis=2)
            c = mindspore.ops.cat(mindspore.ops.chunk(c, 2, axis=0), axis=2)

        init_dec_states = (self._dec_h(h).squeeze(0),
                            self._dec_c(c).squeeze(0))
        return outputs, init_dec_states



    def bs_decode(self, inp, src_vocab,tgt_vocab, bsize=4):
        inp = mindspore.Tensor([[int(word) for word in inp]],dtype=mindspore.int64)
        inp_len = mindspore.Tensor([inp.shape[1]],dtype=mindspore.int64)
        attn_mask = mindspore.ops.ones_like(inp,dtype=mindspore.int64)
        SOS, END = tgt_vocab["<s>"], tgt_vocab["</s>"]

        top_k_scores = mindspore.ops.zeros(bsize)

        top_k_words = mindspore.ops.ones([bsize, 1],dtype=mindspore.int32) * SOS

        completed_seqs = []
        completed_seqs_score = []

        prev_words = top_k_words
        src_vocab_size = len(src_vocab)
        tgt_vocab_size=len(tgt_vocab)
        step = 1
        k = 50#目标生成口令数目

        enc_outs, (h, c) = self.encode(inp, inp_len)
        h = h.broadcast_to((bsize, h.shape[1]))
        c = c.broadcast_to((bsize, h.shape[1]))

        while True:

            dec_out, (h, c) = self.decoder._step(
                prev_words, (h, c), enc_outs, attn_mask)
            logit = mindspore.ops.log_softmax(dec_out, axis=1) #[k, vocab_size]

            logit = top_k_scores.unsqueeze(1).expand_as(logit) + logit

            if step == 1:
                cur_beam = min(logit.shape[1], bsize)
                top_k_scores, ctop_k_words = logit[0].topk(cur_beam, dim=0)
            else:
                cur_beam = min(logit.view(-1).shape[0], bsize)
                top_k_scores, ctop_k_words = logit.view(-1).topk(cur_beam, dim=0)

            pw_inds_tk = ctop_k_words // tgt_vocab_size

            next_word_inds = ctop_k_words % tgt_vocab_size

            top_k_words = mindspore.ops.cat([top_k_words[pw_inds_tk],
                                    next_word_inds.unsqueeze(1)],
                                    axis=1)

            
            incomplete_word_ind = [i for i, word_ind in enumerate(next_word_inds)
                                    if word_ind != tgt_vocab['</s>']]
            complete_word_ind = [ind for ind in range(len(next_word_inds))
                                    if ind not in incomplete_word_ind]

            if len(complete_word_ind):
                completed_seqs.extend(top_k_words[complete_word_ind].tolist())
                completed_seqs_score.extend(top_k_scores[complete_word_ind])
            k -= len(complete_word_ind)
            if k <= 0:
                break

            top_k_words = top_k_words[incomplete_word_ind]
            top_k_scores = top_k_scores[incomplete_word_ind]
            h = h[pw_inds_tk[incomplete_word_ind]]
            c = c[pw_inds_tk[incomplete_word_ind]]
            prev_words = top_k_words[:, -1:]
            if top_k_words.shape[0]==0:
                break
            step += 1
            if step>32:
                break

        SENT = []
        ind=0
        for max_score_seqs in completed_seqs:
            cur_prob=completed_seqs_score[ind]
            SENT.append([max_score_seqs,cur_prob.item()])
            ind+=1
        SENT=sorted(SENT,key=lambda x:x[1],reverse=True)

        return SENT





class AttnDecoder(nn.Cell):
    def __init__(self, embedding, hidden_size,
                 output_size, enc_out_dim, n_layers=1, dropout=0.1):
        super(AttnDecoder, self).__init__()

        self.embedding = embedding
        self.n_layers = n_layers

        emb_size=embedding.embedding_size
        self.decoder_cell = nn.LSTMCell(emb_size, hidden_size)
        self.attn = nn.Dense(enc_out_dim, hidden_size)
        self.concat = nn.Dense(enc_out_dim+hidden_size, hidden_size)
        self.out = nn.Dense(hidden_size, output_size)

    def construct(self, target, init_states, enc_outs, attn_mask):
        max_len = target.shape[1]
        states = init_states
        logits = []
        for i in range(max_len):

            target_i = target[:, i:i+1]

            logit, states = self._step(target_i, states, enc_outs, attn_mask)
            logits.append(logit)
        logits = mindspore.ops.stack(logits, axis=1)

        return logits

    def _step(self, inp, last_hidden, enc_outs, attn_mask):
        embed = self.embedding(inp).squeeze(1)
        # run one step decoding
        h_t, c_t = self.decoder_cell(embed, last_hidden)
        attn_scores = self.get_attn(h_t, enc_outs, attn_mask)
        
        context = attn_scores.matmul(enc_outs.swapaxes(0,1))
        context = context.squeeze(1)

        concat_out = mindspore.ops.tanh(self.concat(
            mindspore.ops.cat([context, h_t], axis=1)
        ))

        logit = mindspore.ops.log_softmax(self.out(concat_out), axis=-1)
        return logit, (h_t, c_t)


    def get_attn(self, dec_out, enc_outs, attn_mask):
        keys = values = enc_outs
        query = dec_out.unsqueeze(0)
        weights = mindspore.ops.sum(query * self.attn(keys), dim=2)
        weights = weights.swapaxes(0, 1)
        weights = weights.masked_fill(attn_mask==0, -1e18)
        weights = weights.unsqueeze(1)
        return mindspore.ops.softmax(weights, axis=2)


def len_mask(lens):
    max_len = max(lens)
    batch_size = len(lens)
    mask=np.zeros((batch_size, max_len),np.int8)
    mask=mindspore.Tensor(mask,dtype=mindspore.int8)
    for i, l in enumerate(lens):
        for j in range(l):
            mask[i,j]=1
    return mask





















    #
