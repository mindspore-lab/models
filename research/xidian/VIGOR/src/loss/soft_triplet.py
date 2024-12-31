# Copyright 2024 Xidian University
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
from mindspore import nn
from mindspore import ops


class SoftTripletBiLoss(nn.Cell):
    def __init__(self, margin=None, alpha=20, **kwargs):
        super(SoftTripletBiLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def construct(self, inputs_q, inputs_k):
        loss_1, mean_pos_sim_1, mean_neg_sim_1 = self.single_forward(inputs_q, inputs_k)
        loss_2, mean_pos_sim_2, mean_neg_sim_2 = self.single_forward(inputs_k, inputs_q)
        return (loss_1+loss_2)*0.5, (mean_pos_sim_1+mean_pos_sim_2)*0.5, (mean_neg_sim_1+mean_neg_sim_2)*0.5

    def single_forward(self, inputs_q, inputs_k):
        n = inputs_q.shape[0]

        normalized_inputs_q = inputs_q / ops.norm(inputs_q, dim=1, keepdim=True)
        normalized_inputs_k = inputs_k / ops.norm(inputs_k, dim=1, keepdim=True)
        # Compute similarity matrix
        sim_mat = ops.matmul(normalized_inputs_q, normalized_inputs_k.t())

        # split the positive and negative pairs
        eyes_ = ops.eye(n)

        pos_mask = eyes_.eq(1)
        neg_mask = ~pos_mask

        pos_sim = ops.masked_select(sim_mat, pos_mask)
        neg_sim = ops.masked_select(sim_mat, neg_mask)

        pos_sim_ = pos_sim.unsqueeze(dim=1).broadcast_to((n, n-1))
        neg_sim_ = neg_sim.reshape(n, n-1)

        loss_batch = ops.log(1 + ops.exp((neg_sim_ - pos_sim_) * self.alpha))

        loss = loss_batch.mean()

        mean_pos_sim = pos_sim.mean().item()
        mean_neg_sim = neg_sim.mean().item()
        return loss, mean_pos_sim, mean_neg_sim
    