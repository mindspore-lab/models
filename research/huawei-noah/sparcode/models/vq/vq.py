# Copyright 2023 Huawei Technologies Co., Ltd
# Copyright (c) 2022-present, Kakao Brain Corp.
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

import numpy as np
import mindspore
from mindspore import nn


class VQEmbedding(nn.Embedding):
    """VQ embedding module with ema update."""

    # 用于密码本的生成
    def __init__(self, K, D, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5):
        super().__init__(K + 1, D, padding_idx=K)
        # K：密码本密码的数量
        # D：密码本的维度
        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.K = K
        self.training = True
        if self.ema:
            # _ = [p.requires_grad_(False) for p in self.trainable_params()]
            _ = [
                mindspore.Parameter(p, name="w2", requires_grad=False)
                for p in self.trainable_params()
            ]

            # padding index is not updated by EMA
            self.weight = mindspore.Parameter(
                mindspore.Tensor(np.zeros((K + 1, D)), mindspore.float32),
                name="weight",
                requires_grad=False,
            )
            self.cluster_size_ema = mindspore.Parameter(
                mindspore.Tensor(np.zeros(([1, 256])), mindspore.float32),
                name="w",
                requires_grad=False,
            )
            # self.register_buffer('cluster_size_ema', mindspore.ops.zeros(K))
            self.embed_ema = mindspore.Parameter(
                self.weight[:-1, :], name="w1", requires_grad=False
            )
            # self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())

    def compute_distances(self, inputs):
        # 计算距离
        codebook_t = self.weight[:-1, :].t()

        (D, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == D

        inputs_flat = inputs.reshape(-1, D)

        inputs_norm_sq = mindspore.ops.sum(inputs_flat.pow(2.0), dim=1, keepdim=True)
        codebook_t_norm_sq = mindspore.ops.sum(codebook_t.pow(2.0), dim=0, keepdim=True)
        distances = mindspore.ops.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-100.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)  # [B, h, w, K or K+1]
        return distances

    def find_nearest_embedding(self, inputs):
        # 查找最相似的emb并返回索引
        distances = self.compute_distances(inputs)  # [B, h, w, K or K+1]
        embed_idxs = distances.argmin(axis=-1)  # use padding index or not

        return embed_idxs

    def _tile_with_noise(self, x, target_n):
        B, D = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(D) * 0.01 / np.sqrt(D)
        x = x.repeat(n_repeats, 1)
        x = x + mindspore.ops.rand_like(x) * std
        return x

    def _update_buffers(self, vectors, idxs):
        K, D = self.weight.shape[0] - 1, self.weight.shape[-1]

        vectors = vectors.reshape(-1, D)
        idxs = idxs.reshape(-1)

        n_vectors = vectors.shape[0]
        n_total_embed = K
        # print('vectors',vectors)
        one_hot_idxs = vectors.new_zeros((n_total_embed, n_vectors))

        one_hot_idxs = mindspore.ops.tensor_scatter_elements(
            input_x=one_hot_idxs,
            axis=0,
            indices=idxs.unsqueeze(0),
            updates=vectors.new_ones((1, n_vectors)),
        )
        cluster_size = one_hot_idxs.sum(axis=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors
        self.cluster_size_ema = (self.cluster_size_ema * self.decay) + (
            cluster_size * (1 - self.decay)
        )
        self.embed_ema = (self.embed_ema * self.decay) + (
            vectors_sum_per_cluster * (1 - self.decay)
        )

        if self.restart_unused_codes:
            if n_vectors < K:
                vectors = self._tile_with_noise(vectors, K)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[mindspore.ops.randperm(n_vectors)][:K]

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema = (self.embed_ema * usage) + (_vectors_random * (1 - usage))
            self.cluster_size_ema = self.cluster_size_ema * usage.view(-1)
            self.cluster_size_ema = self.cluster_size_ema + (
                mindspore.ops.ones_like(self.cluster_size_ema) * (1 - usage).view(-1)
            )

    def _update_embedding(self):
        K = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + K * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def construct(self, inputs, mode="st"):
        # 密码本的生成
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)

        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()
        embeds1 = self.embed(embed_idxs)
        # commitment_loss = self.compute_commitment_loss(inputs, embeds)
        return (embeds, embeds1, embed_idxs)

    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list = []
        for idx, quant in enumerate(quant_list):
            partial_loss = (x - quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)

        commitment_loss = mindspore.ops.mean(mindspore.ops.stack(loss_list))
        return commitment_loss

    def embed(self, idxs):
        embeds = super().construct(idxs)
        return embeds
