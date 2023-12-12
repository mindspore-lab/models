# Copyright 2023 Huawei Technologies Co., Ltd
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


import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import Normal, initializer


class BaseModel(nn.Cell):
    def __init__(self, n_items, n_cate, item_cate_map, seq_len, embedding_size, l2_reg):
        super().__init__()
        self.n_items = n_items + 1
        self.n_cate = n_cate + 1
        self.item_cate_map = mindspore.Parameter(
            mindspore.tensor(item_cate_map), requires_grad=False
        )
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.item_embedding.weight = initializer(
            Normal(sigma=1, mean=0.0),
            shape=[self.n_items, self.embedding_size],
            dtype=mindspore.float32,
        )
        self.cate_embedding = nn.Embedding(
            self.n_cate, self.embedding_size, padding_idx=0
        )
        self.cate_embedding.weight = initializer(
            Normal(sigma=1, mean=0.0),
            shape=[self.n_cate, self.embedding_size],
            dtype=mindspore.float32,
        )
        # Normal(self.cate_embedding.weight, mean=0, std=1.0e-4)
        self.device = "cpu"

    def user_encoder(self, hist_items):
        return NotImplementedError

    def item_encoder(self, item):
        item_embedding = self.item_embedding(item)
        return item_embedding

    def bce_loss(self, pos_logits, neg_logits):
        bce_criterion = mindspore.nn.BCEWithLogitsLoss()
        pos_labels, neg_labels = mindspore.ops.ones(
            pos_logits.shape
        ), mindspore.ops.zeros(neg_logits.shape)
        loss = bce_criterion(pos_logits, pos_labels)
        loss += bce_criterion(neg_logits, neg_labels)
        return loss

    def l2_loss(self, hist_items):
        hist_items = self.item_embedding(hist_items)
        emb_loss = mindspore.ops.norm(hist_items) ** 2
        return self.l2_reg * emb_loss / self.seq_len

    def softmax_loss(self, pos_logits, neg_logits):
        logits = mindspore.ops.concat(
            [pos_logits.view(-1, 1), neg_logits], axis=1
        )  # (B,2)
        softmax_logits = nn.LogSoftmax(axis=1)(logits)
        softmax_loss = mindspore.ops.mean(-softmax_logits[:, 0])
        return softmax_loss

    def get_cate_embedding(self, items):
        if len(items.shape) == 2:
            batch_size, seq_len = items.shape
        cate = self.item_cate_map[items.view(-1)]
        cate_embedding = self.cate_embedding(cate)  # (B, N, D)
        if len(items.shape) == 2:
            cate_embedding = cate_embedding.view(batch_size, seq_len, -1)
        return cate_embedding

    def get_item_embedding(self):
        all_items = mindspore.tensor(mindspore.ops.arange(0, self.n_items)).long()
        item_embedding = self.item_encoder(all_items)
        return item_embedding

    def get_user_embedding(self, hist_items):
        hist_items = mindspore.tensor(hist_items).long()
        user_embedding = self.user_encoder(hist_items)
        return user_embedding

    @staticmethod
    def set_requires_grad(params, requires_grad):
        for p in params:
            p.requires_grad = requires_grad
