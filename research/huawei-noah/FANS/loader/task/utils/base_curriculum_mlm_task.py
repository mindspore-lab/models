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


import random
from abc import ABC
from typing import Optional, Union, Type

import mindspore
from pigmento import pnt

from loader.task.base_loss import TaskLoss
from loader.task.utils.base_mlm_task import BaseMLMTask, MLMBertBatch


class CurriculumMLMBertBatch(MLMBertBatch):
    def __init__(self, batch):
        super(CurriculumMLMBertBatch, self).__init__(batch=batch)
        self.weight = None  # type: Optional[float]

        self.register('weight')


class BaseCurriculumMLMTask(BaseMLMTask, ABC):
    name = 'base-curriculum-mlm'
    batcher: Union[Type[CurriculumMLMBertBatch]]

    def __init__(
            self,
            weighted=False,
            curriculum_steps=10,
            weight_decay=1,
            weight_center=None,
            **kwargs
    ):
        """

        :param curriculum_steps: 课程学习总步长
        :param weighted: 是否使用加权模式（如当前过了40%的epoch，则mask 40%的数据权重为1）
        :param weight_decay: 加权模式下的权重衰减

        """
        super(BaseCurriculumMLMTask, self).__init__(**kwargs)
        self.curriculum_steps = curriculum_steps  # 10
        self.current_mask_ratio = 0

        self.weighted = weighted
        self.weight_decay = weight_decay
        self.weight_center = weight_center

    def start_epoch(self, current_epoch, total_epoch):  # 3 50
        if self.weighted:
            self.current_mask_ratio = (current_epoch + 1) * 1.0 / total_epoch
        else:
            self.current_mask_ratio = \
                (int(current_epoch * self.curriculum_steps // total_epoch) + 1) * 1.0 / self.curriculum_steps
        pnt(f'set current mask ratio to {self.current_mask_ratio}')

    def prepare_batch(self, batch: CurriculumMLMBertBatch):
        super().prepare_batch(batch)
        if self.weighted:
            mask_ratio = random.random()
            weight_center = self.current_mask_ratio if self.weight_center is None else self.weight_center
            batch.weight = (1 - abs(weight_center - mask_ratio)) * self.weight_decay
        else:
            mask_ratio = self.current_mask_ratio
            batch.weight = 1
        batch.mask_ratio = mask_ratio
        if not self.is_training:
            batch.weight = 1

    def calculate_loss(self, batch: CurriculumMLMBertBatch, output, **kwargs) -> TaskLoss:
        return super().calculate_loss(
            batch=batch,
            output=output,
            weight=batch.weight,
            **kwargs
        )

    def _test__curriculum(self, indexes, mask_labels_col, output, metric_pool, col_name):
        output = output[self.depot.cols[col_name].voc.name]
        col_mask = mask_labels_col[col_name]

        for i_batch in range(len(indexes)):
            argsorts = []
            for i_tok in range(self.dataset.max_sequence):
                if col_mask[i_batch][i_tok]:
                    argsorts.append(
                        mindspore.ops.argsort(
                            output[i_batch][i_tok], descending=True).cpu().tolist()[:metric_pool.max_n])
                else:
                    argsorts.append(None)

            ground_truth = self.depot.pack_sample(indexes[i_batch])[col_name]
            candidates = []
            # candidates_set = set()
            for depth in range(metric_pool.max_n):
                for i_tok in range(self.dataset.max_sequence):
                    if col_mask[i_batch][i_tok]:
                        if argsorts[i_tok][depth] not in candidates:
                            candidates.append(argsorts[i_tok][depth])
                if len(candidates) >= metric_pool.max_n:
                    break

            metric_pool.push(candidates, ground_truth)

    def test__curriculum(self, batch, output, metric_pool):
        raise NotImplementedError
