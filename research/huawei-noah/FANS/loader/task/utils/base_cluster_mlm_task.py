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


import json
import os.path
from abc import ABC
from typing import Union

import mindspore
# from UniTok import Vocab
from mindspore import nn
# from UniTok import Vocab
from loader.task.base_loss import TaskLoss
from loader.task.utils.base_classifiers import BertClusterClassifier, BertClassifier

from loader.task.utils.base_mlm_task import BaseMLMTask, MLMBertBatch


class ClusterMLMTaskLoss(TaskLoss):
    def __init__(self, local_loss, cluster_loss):
        super(ClusterMLMTaskLoss, self).__init__(loss=local_loss + cluster_loss)
        self.local_loss = local_loss
        self.cluster_loss = cluster_loss

    def backward(self):
        loss = self.loss + self.cluster_loss
        return loss




class BaseClusterMLMTask(BaseMLMTask, ABC):
    name = 'base-cluster-mlm'
    cluster_cls_module: Union[BertClusterClassifier]
    cls_module: Union[BertClassifier]

    def __init__(
            self,
            cluster_json,
            k_global='k_global',
            p_global='p_global',
            k_local='k_local',
            p_local='p_local',
            k_cluster='k_cluster',
            p_cluster='p_cluster',
            grad_cluster_loss=True,
            grad_local_loss=True,
            **kwargs
    ):
        super(BaseClusterMLMTask, self).__init__(**kwargs)

        self.k_global = k_global
        self.k_local = k_local
        self.k_cluster = k_cluster
        self.p_global = p_global
        self.p_local = p_local
        self.p_cluster = p_cluster
        self.grad_cluster_loss = grad_cluster_loss
        self.grad_local_loss = grad_local_loss

        self.cluster_json = cluster_json
        self.col_cluster_dict = {
            self.k_global: self.k_cluster,
            self.p_global: self.p_cluster
        }

        self.col_pairs = [(self.k_cluster, self.k_local), (self.p_cluster, self.p_local)]


    def init(self, **kwargs):
        super().init(**kwargs)
        self.cluster_vocab_count = json.load(open(os.path.join('data/ListContUni/aotm-n10', self.cluster_json)))
        self.n_clusters = 10

    def get_embedding(self, **kwargs):
        return super().get_embedding(**kwargs, enable_attrs={self.k_global, self.p_global})

    def _init_extra_module(self):
        return nn.CellDict(dict(
            cluster_cls=self.cluster_cls_module.create(
                key='cluster',
                cluster_vocabs=self.cluster_vocab_count,
                config=self.model_init.model_config,
            ),
            cls=self.cls_module.create(
                config=self.model_init.model_config,
                key='cluster_id',
                vocab_size=10,
            )
        ))

    def _produce_output(self, last_hidden_state, batch: MLMBertBatch,test):
        mask_labels = batch.mask_labels  # type:

        output_dict = dict()

        cls_module = self.extra_module['cls']
        cluster_cls_module = self.extra_module['cluster_cls']
        output_dict['pred_cluster_distribution'] = pred_clusters = cls_module(last_hidden_state)  # [B, N, V]
        output_dict['pred_cluster_labels'] = pred_cluster_labels = mindspore.ops.argmax(pred_clusters, dim=-1)
        for col_name, local_col_name in self.col_pairs:
            if col_name not in batch.mask_labels_col:
                continue
            mask_labels_col = batch.mask_labels_col[col_name]
            mask_labels_col1 = mask_labels_col.asnumpy().tolist()
            col_mask = batch.col_mask[col_name]
            col_mask1 = col_mask.asnumpy().tolist()
            masked_elements = mindspore.ops.not_equal(mindspore.tensor(mask_labels_col1,dtype=mindspore.int32), mindspore.tensor(col_mask1,dtype=mindspore.int32))
            if not test:
                current_cluster_labels = masked_elements * (mask_labels + 1)
                current_pred_cluster_labels = masked_elements * (pred_cluster_labels + 1)
                cluster_labels = mindspore.ops.equal(current_cluster_labels, current_pred_cluster_labels) * current_cluster_labels - 1
            else:
                cluster_labels = masked_elements * (pred_cluster_labels + 1) - 1
            output_dict[local_col_name] = cluster_cls_module(
                last_hidden_state,
                cluster_labels=cluster_labels,
            )
        return output_dict

    def calculate_loss(self, batch: MLMBertBatch, output, **kwargs) -> ClusterMLMTaskLoss:
        weight = kwargs.get('weight', 1)
        mask_labels = batch.mask_labels # type:

        total_cluster_loss = mindspore.Tensor(0, dtype=mindspore.float32)
        total_local_loss = mindspore.Tensor(0, dtype=mindspore.float32)
        # self.col_pairs = [(self.k_cluster, self.k_local), (self.p_cluster, self.p_local)]
        for col_name, local_col_name in self.col_pairs:
            if col_name not in batch.mask_labels_col:
                continue
            vocab_size = 10
            mask_labels_col = batch.mask_labels_col[col_name]  # type

            col_mask = batch.col_mask[col_name]
            masked_elements = mindspore.ops.not_equal(col_mask, mask_labels_col)  # type:
            if not mindspore.ops.sum(masked_elements):
                continue

            if self.grad_cluster_loss:
                # print('11111111111111')
                distribution = mindspore.ops.masked_select(
                    output['pred_cluster_distribution'], masked_elements.unsqueeze(dim=-1)
                ).view(-1, vocab_size)
                col_labels = mindspore.ops.masked_select(mask_labels, masked_elements)
                col_labels = mindspore.ops.cast(col_labels,mindspore.int32)
                loss = self.loss_fct(
                    distribution,
                    col_labels
                )
                total_cluster_loss += loss * weight

            if self.grad_local_loss:
                cluster_labels = masked_elements * (mask_labels + 1)
                pred_cluster_labels = masked_elements * (output['pred_cluster_labels'] + 1)
                cluster_labels = mindspore.ops.eq(cluster_labels, pred_cluster_labels) * cluster_labels - 1
                local_labels = batch.attr_ids[col_name][local_col_name]
                for i_cluster in range(self.n_clusters):
                    if not (cluster_labels == i_cluster).sum():
                        continue

                    cluster_masked_elements = ((cluster_labels == i_cluster) * masked_elements.float())
                    i_cluster_labels = mindspore.ops.masked_select(local_labels.float(), cluster_masked_elements.bool())  # [B, K]
                    loss = self.loss_fct(
                        output[local_col_name][i_cluster],  # [B, K, V]
                        i_cluster_labels.int(),  # [B, K]
                    )
                    total_local_loss += loss * weight / self.n_clusters
        # exit(0)

        return total_local_loss + total_cluster_loss


    def test__curriculum(self, batch: MLMBertBatch, output, metric_pool,local_global_maps):
        indexes =batch.append_info
        pred_cluster_labels = output['pred_cluster_labels']
        output = output[self.p_local]
        col_mask = batch.mask_labels_col[self.p_cluster]
        cluster_indexes = [0] * 10
        for i_batch in range(len(indexes)):
            arg_sorts = []
            for i_tok in range(103):
                if col_mask[i_batch][i_tok]:
                    cluster_id = pred_cluster_labels[i_batch][i_tok]
                    top_items = mindspore.ops.argsort(
                        output[cluster_id][cluster_indexes[cluster_id]], descending=True
                    ).asnumpy().tolist()[:metric_pool.max_n]
                    top_items = [local_global_maps[cluster_id][item] for item in top_items]
                    arg_sorts.append(top_items)
                    cluster_indexes[cluster_id] += 1
                else:
                    arg_sorts.append(None)
            pack = batch.attr_ids['p_cluster'][self.p_global]
            zeros = mindspore.ops.zeros_like(pack)
            greater = mindspore.ops.Greater()
            output1 = greater(pack, zeros)
            ground_truth = pack[output1].asnumpy().tolist()
            candidates = []
            for depth in range(metric_pool.max_n):
                for i_tok in range(103):
                    if col_mask[i_batch][i_tok]:
                        if arg_sorts[i_tok][depth] not in candidates:
                            candidates.append(arg_sorts[i_tok][depth])
                if len(candidates) >= metric_pool.max_n:
                    break
            metric_pool.push(candidates, ground_truth)

