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


import itertools
from tqdm import tqdm
import mindspore
import mindspore.nn as nn
from models.BaseModel import BaseModel
from models.layers import MLP_Layer, SELayer
from models.vq import DVQEmbedding


class DQMatch(BaseModel):
    def __init__(
        self,
        n_items,
        n_cate,
        item_cate_map,
        seq_len,
        embedding_size,
        l2_reg,
        dropout,
        k_c=1024,
        k_u=10,
        k_i=1,
        token_embedding_size=64,
        beta=1,
        batch_norm=False,
        num_codebooks=2,
    ):
        super(DQMatch, self).__init__(
            n_items, n_cate, item_cate_map, seq_len, embedding_size, l2_reg
        )
        """Use EMA to update codebook """
        self.token_embedding_size = token_embedding_size
        self.K_c = k_c
        self.K_u = k_u
        self.K_i = k_i
        self.beta = beta
        self.num_codebooks = num_codebooks
        """ codebook """
        self.use_codebook_ema = True
        self.codebook = DVQEmbedding(
            self.num_codebooks,
            self.K_c,
            self.token_embedding_size,
            ema=self.use_codebook_ema,
        )
        if self.use_codebook_ema:
            self.set_requires_grad(self.codebook.trainable_params(), False)
        self.se_layers = nn.CellList(
            [SELayer(channel=self.seq_len, reduction=2) for _ in range(self.K_u)]
        )
        self.user_linear = nn.CellList(
            [
                nn.Dense(
                    in_channels=embedding_size, out_channels=self.token_embedding_size
                )
                for _ in range(self.K_u)
            ]
        )
        self.item_linear = nn.CellList(
            [
                nn.Dense(
                    in_channels=embedding_size, out_channels=self.token_embedding_size
                )
                for _ in range(self.K_i)
            ]
        )
        """ Late Interaction """
        self.late_interaction = MLP_Layer(
            input_dim=self.token_embedding_size + self.K_i * self.token_embedding_size,
            output_dim=1,
            hidden_units=[256, 256, 256],
            hidden_activations="ReLU",
            output_activation=None,
            dropout_rates=dropout,
            batch_norm=batch_norm,
            use_bias=True,
        )

    def item_token_encoder(self, item):
        # (B,) --> (B,D)
        item_embedding = self.item_encoder(item)
        item_output = [
            self.item_linear[i](item_embedding) for i in range(len(self.item_linear))
        ]
        if len(item_embedding.shape) == 2:
            item_output = mindspore.ops.stack(item_output, axis=1)  # B, K_i, D
        elif len(item_embedding.shape) == 3:
            item_output = mindspore.ops.stack(item_output, axis=2)  # B, Neg_Num, K_i, D
        return item_output

    def quantization(self, z_e_x):
        z_q_x_st, z_q_x, indices = self.codebook(z_e_x, mode="st")
        return z_q_x_st, z_q_x, z_e_x

    def get_logits(self, user_token_embedding, item_token_embedding):
        item_token_embedding = item_token_embedding.view(
            -1, 1, self.K_i * self.token_embedding_size
        )  # (B, 1, K_i*D)
        item_token_embedding = mindspore.ops.repeat_interleave(
            item_token_embedding, repeats=user_token_embedding.shape[1], axis=1
        )  # (B, K_u, K_i*D)
        interaction_input = mindspore.ops.concat(
            [user_token_embedding, item_token_embedding], axis=2
        )  # B,K_u, D+K_i*D
        logits = self.late_interaction(interaction_input)  # B,K_u,1
        return logits.view(logits.shape[0], logits.shape[1])

    def get_token_logits(self):
        logits = dict()
        user_token_embedding = self.codebook.get_codebook()  # num_codebook, K_c,D//2
        item_token_embedding = self.item_token_encoder(
            mindspore.tensor([i for i in range(self.n_items)]).int()
        )  # n_items, K_i, D
        item_token_embedding = item_token_embedding.view(
            -1, item_token_embedding.size(1) * item_token_embedding.size(2)
        )
        for x in tqdm(
            itertools.product([i for i in range(self.K_c)], repeat=self.num_codebooks)
        ):
            u_token = mindspore.ops.cat(
                [user_token_embedding[k][t] for k, t in enumerate(x)], axis=0
            ).view(1, -1)
            u_token = mindspore.ops.repeat_interleave(
                u_token, repeats=item_token_embedding.size(0), axis=0
            )  # (n_items,D)
            interaction_input = mindspore.ops.concat(
                [u_token, item_token_embedding], axis=1
            )  # n_items, D+K_i*D
            u_logits = self.late_interaction(interaction_input)  # n_items
            # u_logits = u_logits.cpu().numpy()
            logits[x] = u_logits.view(-1).cpu()
        return logits

    def get_user_token(self, hist_items):
        user_output = self.user_encoder(hist_items)  # B, K_u, D
        user_token = self.codebook(user_output)
        return user_token

    def construct(self, input_dict):
        hist_items, pos_items, neg_items = (
            input_dict["hist_items"],
            input_dict["pos_items"],
            input_dict["neg_items"],
        )
        user_embedding = self.user_encoder(hist_items)  # B, K_u, D
        # z_q_x_st for the next network, z_q_x_st.shape: (B,K_u,D)
        z_q_x_st, z_q_x, z_e_x = self.quantization(user_embedding)
        pos_item_embedding = self.item_token_encoder(pos_items)  # B,K_i, D
        neg_item_embedding = self.item_token_encoder(neg_items)  # B,Num_Neg,K_i,D
        pos_logits = self.get_logits(z_q_x_st, pos_item_embedding)  # B, K_u
        pos_logits = mindspore.ops.sum((pos_logits), dim=1)  # B,
        neg_logits = [
            mindspore.ops.sum(
                self.get_logits(z_q_x_st, neg_item_embedding[:, i, :, :]), dim=1
            )
            for i in range(neg_item_embedding.shape[1])
        ]
        neg_logits = mindspore.ops.stack(neg_logits, axis=1)  # B, Num_Neg
        softmax_loss = self.softmax_loss(pos_logits, neg_logits)
        reg_loss = self.l2_loss(hist_items)
        if self.use_codebook_ema:
            vq_loss = mindspore.ops.mse_loss(z_e_x, z_q_x)
        else:
            vq_loss = mindspore.ops.mse_loss(
                z_q_x, z_e_x
            ) + 0.25 * mindspore.ops.mse_loss(z_e_x, z_q_x)
        loss = softmax_loss + reg_loss + self.beta * vq_loss
        return loss, vq_loss

    def test_forward(self, input_dict):
        hist_items, items = input_dict["hist_items"], input_dict["items"]
        user_embedding = self.user_encoder(hist_items)
        z_q_x_st, z_q_x, z_e_x = self.quantization(user_embedding)
        item_embedding = self.item_token_encoder(items)  # B,Num_Neg,K_i,D
        logits = mindspore.ops.sum(self.get_logits(z_q_x_st, item_embedding), dim=1)
        return logits


class DQMatchWithDNN(DQMatch):
    def __init__(
        self,
        n_items,
        n_cate,
        item_cate_map,
        seq_len,
        embedding_size,
        l2_reg,
        dropout,
        k_c=1024,
        k_u=10,
        k_i=1,
        token_embedding_size=64,
        beta=1,
        batch_norm=False,
        num_codebooks=2,
    ):
        super(DQMatchWithDNN, self).__init__(
            n_items,
            n_cate,
            item_cate_map,
            seq_len,
            embedding_size,
            l2_reg,
            dropout,
            k_c,
            k_u,
            k_i,
            token_embedding_size,
            beta,
            batch_norm,
            num_codebooks,
        )
        self.dnn = MLP_Layer(
            input_dim=embedding_size,
            output_dim=embedding_size,
            hidden_units=[256, 256, 256],
            hidden_activations="ReLU",
            output_activation=None,
            dropout_rates=dropout,
            batch_norm=False,
            use_bias=True,
        )

    def user_encoder(self, hist_items):
        hist_items_embedding = self.item_encoder(hist_items)  # B, L, D
        mask = mindspore.ops.where(
            hist_items == 0,
            mindspore.ops.zeros_like(hist_items),
            mindspore.ops.ones_like(hist_items),
        )  # B, L
        hist_items_embedding = mindspore.ops.sum(
            hist_items_embedding, dim=1
        ) / mindspore.ops.sum(mask, dim=1).view(-1, 1)
        user_output = self.dnn(hist_items_embedding)
        user_output = [self.user_linear[i](user_output) for i in range(self.K_u)]
        user_output = mindspore.ops.stack(user_output, axis=1)
        return user_output
