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
from .vq import VQEmbedding


class DVQEmbedding(nn.Cell):
    def __init__(self, num, K, D, ema):
        super(DVQEmbedding, self).__init__()
        assert (
            D % num == 0
        ), "D must be divided by num withour remainder, while D={} and num={}".format(
            D, num
        )
        self.num = num  # the number of codebooks.
        self.D = D
        self.codebooks = nn.CellList(
            [VQEmbedding(K, D // num, ema) for _ in range(num)]
        )

    def construct(self, z_e_x, mode=""):
        if mode == "st":
            parts = z_e_x.split(self.D // self.num, axis=-1)
            z_q_x_st_list, z_q_x_list, indices_list = [], [], []
            for i, part in enumerate(parts):
                z_q_x_st, z_q_x, indices = self.codebooks[i](part, "st")
                z_q_x_st_list.append(z_q_x_st)
                z_q_x_list.append(z_q_x)
                indices_list.append(indices)
            return (
                mindspore.ops.cat(z_q_x_st_list, axis=-1),
                mindspore.ops.cat(z_q_x_list, axis=-1),
                mindspore.ops.stack(indices_list, axis=-1),
            )
        else:
            parts = z_e_x.split(self.D // self.num, dim=-1)
            latents = []
            for i, part in enumerate(parts):
                latent = self.codebooks[i](part)
                latents.append(latent)
            return mindspore.ops.stack(latents, axis=-1)

    def get_codebook(self):
        codebook = []
        for i in range(self.num):
            codebook.append(self.codebooks[i].get_codebook())
        codebook = mindspore.ops.stack(codebook, axis=0)
        return codebook
