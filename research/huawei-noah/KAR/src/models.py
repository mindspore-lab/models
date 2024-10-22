# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""

-*- coding: utf-8 -*-
@File  : models.py
"""

import mindspore as ms
from mindspore import nn
from mindspore import ops as P
from mindspore.common.initializer import TruncatedNormal
ms.set_context(mode=ms.PYNATIVE_MODE)


class MLP(nn.Cell):
    def __init__(self, inp_dim, dropout, layers):
        super(MLP, self).__init__()
        self.batch_norm = nn.BatchNorm1d(inp_dim)
        self.reshape = P.Reshape()
        self.layers = [inp_dim] + layers  # [224, 256, 64]
        self.layer_num = len(layers)   # 2
        seq = [[nn.Dense(self.layers[i], self.layers[i + 1]), nn.ReLU(), nn.Dropout(p=dropout)]
               for i in range(self.layer_num)]
        self.seq = nn.SequentialCell(sum(seq, []))

    def construct(self, inp):
        inp_shape = inp.shape
        inp = self.reshape(inp, (-1, inp_shape[-1]))
        batch_norm = self.batch_norm(inp)
        fc_out = self.seq(batch_norm)
        fc_out = self.reshape(fc_out, inp_shape[:-1] + (self.layers[-1],))
        return fc_out


class AttentionPooling(nn.Cell):
    def __init__(self, inp_dim, dropout, layers):
        super(AttentionPooling, self).__init__()
        self.layers = [inp_dim * 4] + layers  # [224, 256, 64]
        self.layer_num = len(layers)   # 2
        seq = [[nn.Dense(self.layers[i], self.layers[i + 1]), nn.ReLU(), nn.Dropout(p=dropout)]
               for i in range(self.layer_num)]
        seq = sum(seq, [])
        seq.append(nn.Dense(layers[-1], 1))
        self.fc = nn.SequentialCell(seq)
        self.concat = P.Concat(axis=1)
        self.concat2 = P.Concat(axis=-1)
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.expand_dims = P.ExpandDims()
        self.ones_likes = P.OnesLike()
        self.shape = P.Shape()
        self.softmax = P.Softmax()
        self.squeeze = P.Squeeze(axis=2)

    def construct(self, query, user_seq, mask=None):
        query = self.expand_dims(query, 1)
        seq_len = self.shape(user_seq)[1]
        queries = self.concat([query for _ in range(seq_len)])
        attn_inp = self.concat2([queries, user_seq, queries - user_seq,
                                 queries * user_seq])
        # print(attn_inp.shape)
        attns = self.squeeze(self.fc(attn_inp))
        # print('attns', attns.shape, mask.shape)
        paddings = self.ones_likes(attns) * (-2 ** 32 + 1)
        attns = ms.numpy.where(mask == 0, paddings, attns)
        attns = self.softmax(attns)
        # print('attns', attns.shape)
        out = user_seq * self.expand_dims(attns, 2)
        # print('out', out.shape)
        output = self.reduce_sum(out, axis=1)
        # print('output', output.shape)
        return output, attns


class HEA(nn.Cell):
    def __init__(self, hea_arch, inp_dim, dropout):
        super(HEA, self).__init__()
        share_expt_num, spcf_expt_num, expt_arch, task_num = hea_arch
        self.share_expt_net = nn.CellList([MLP(inp_dim, dropout, expt_arch)
                                           for _ in range(share_expt_num)])
        self.spcf_expt_net = nn.CellList([nn.CellList([MLP(inp_dim, dropout, expt_arch)
                                                       for _ in range(spcf_expt_num)])
                                          for _ in range(task_num)])
        self.gate_net = nn.CellList([nn.Dense(inp_dim, share_expt_num + spcf_expt_num)
                                     for _ in range(task_num)])
        self.stack1 = P.Stack(axis=1)
        self.stack2 = P.Stack(axis=2)
        self.concat = P.Concat(axis=-1)
        self.concat2 = P.Concat(axis=2)
        self.softmax = P.Softmax(axis=-1)
        self.squeeze1 = P.Squeeze(axis=1)
        self.squeeze2 = P.Squeeze(axis=2)
        self.expand_dims = P.ExpandDims()
        self.bat_mat_mul = P.BatchMatMul()
        self.split = P.Split(axis=1, output_num=task_num)

    def construct(self, inp):
        gates = [net(x) for net, x in zip(self.gate_net, inp)]
        gates = self.stack1(gates)
        gates = self.expand_dims(self.softmax(gates), 2)
        cat_x = self.stack1(inp)
        # print('cat_x', cat_x.shape)
        share_experts = [net(cat_x) for net in self.share_expt_net]
        share_experts = self.stack2(share_experts)
        spcf_experts = [self.stack1([net(x) for net in nets])
                        for nets, x in zip(self.spcf_expt_net, inp)]
        spcf_experts = self.stack1(spcf_experts)
        experts = self.concat2([share_experts, spcf_experts])
        expert_mix = self.bat_mat_mul(gates, experts)
        expert_mix = self.squeeze2(expert_mix)
        expert_mix = self.split(expert_mix)
        out = [self.squeeze1(x) for x in expert_mix]
        out = self.concat(out)
        return out


class KAR(nn.Cell):
    """KAR based on DIN"""

    def __init__(self, args, dataset):
        super(KAR, self).__init__()
        self.args = args
        self.augment_num = 2 if args.augment else 0
        args.augment_num = self.augment_num

        self.item_num = dataset.item_num
        self.attr_num = dataset.attr_num
        self.attr_fnum = dataset.attr_ft_num
        self.rating_num = dataset.rating_num
        self.dense_dim = dataset.dense_dim
        self.max_hist_len = args.max_hist_len

        self.embed_dim = args.embed_dim
        self.final_mlp_arch = args.final_mlp_arch
        self.dropout = args.dropout
        self.convert_dropout = args.convert_dropout
        self.din_mlp = args.din_mlp

        self.item_fnum = 1 + self.attr_fnum
        self.hist_fnum = 2 + self.attr_fnum
        self.itm_emb_dim = self.item_fnum * self.embed_dim
        self.hist_emb_dim = self.hist_fnum * self.embed_dim
        self.dens_vec_num = 0
        self.item_embedding = nn.Embedding(self.item_num + 1, self.embed_dim,
                                           embedding_table=TruncatedNormal(sigma=1.0))
        self.attr_embedding = nn.Embedding(self.item_num + 1, self.embed_dim,
                                           embedding_table=TruncatedNormal(sigma=1.0))
        self.rating_embedding = nn.Embedding(self.item_num + 1, self.embed_dim,
                                             embedding_table=TruncatedNormal(sigma=1.0))
        if self.augment_num:
            hea_arch = args.expert_num, args.specific_expert_num, \
                       args.convert_arch, args.augment_num
            # print('dense num', self.dense_dim)
            self.convert_module = HEA(hea_arch, self.dense_dim, self.convert_dropout)
            self.dense_vec_num = args.convert_arch[-1] * self.augment_num
        self.module_inp_dim = self.itm_emb_dim * 2 + self.dense_vec_num
        self.map_layer = nn.Dense(self.hist_emb_dim, self.itm_emb_dim)
        self.attention = AttentionPooling(self.itm_emb_dim, self.dropout, self.din_mlp)
        self.final_mlp = MLP(self.module_inp_dim, self.dropout, self.final_mlp_arch)
        self.final_fc = nn.Dense(self.final_mlp_arch[-1], 1, activation=nn.Sigmoid())
        self.reshape = P.Reshape()
        self.concat = P.Concat(axis=-1)
        self.shape = P.Shape()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()

    def sequence_mask(self, inp, max_len):
        batch_num = self.shape(inp)[0]
        rang = self.tile(self.reshape(P.arange(max_len), (1, max_len)), (batch_num, 1))
        mask = rang < self.expand_dims(inp, -1)
        return mask.astype(ms.float32)

    def construct(self, iid, aid, hist_item_seq, hist_attri_seq, hist_rating_seq,
                  hist_seq_len, item_aug_vec, hist_aug_vec):
        """construct"""
        hist_item_emb = self.reshape(self.item_embedding(hist_item_seq),
                                     (-1, self.max_hist_len, self.embed_dim))
        hist_attr_emb = self.reshape(self.item_embedding(hist_attri_seq),
                                     (-1, self.max_hist_len, self.embed_dim * self.attr_fnum))
        hist_rating_emb = self.reshape(self.item_embedding(hist_rating_seq),
                                       (-1, self.max_hist_len, self.embed_dim))
        hist_emb = self.concat([hist_item_emb, hist_attr_emb, hist_rating_emb])

        iid_emb = self.item_embedding(iid)
        attr_emb = self.reshape(self.attr_embedding(aid),
                                (-1, self.embed_dim * self.attr_fnum))
        item_emb = self.concat([iid_emb, attr_emb])

        mask = self.sequence_mask(hist_seq_len, self.max_hist_len)
        user_behavior = self.map_layer(hist_emb)
        # print(item_emb.shape, user_behavior.shape, self.itm_emb_dim)
        user_interest, _ = self.attention(item_emb, user_behavior, mask)
        # print('user interest', user_interest.shape, user_interest.dtype)
        # print('item emb', item_emb.shape, item_emb.dtype)

        if self.augment_num:
            dens_vec = self.convert_module([hist_aug_vec, item_aug_vec])
            # print('dense vec', dens_vec.shape, dens_vec.dtype)
            concat_input = self.concat([user_interest, item_emb, dens_vec])
        else:
            concat_input = self.concat([user_interest, item_emb])
        # print('concat input', concat_input.shape, self.module_inp_dim)

        mlp_out = self.final_mlp(concat_input)
        preds = self.final_fc(mlp_out)
        preds = self.reshape(preds, (-1,))
        return preds


class KARWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(KARWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, iid, attri_id, hist_item_seq, hist_attri_seq, hist_rating_seq,
                  hist_seq_len, item_aug_vec, hist_aug_vec, label):
        out = self._backbone(iid, attri_id, hist_item_seq, hist_attri_seq, hist_rating_seq,
                             hist_seq_len, item_aug_vec, hist_aug_vec)
        return self._loss_fn(out, label)
