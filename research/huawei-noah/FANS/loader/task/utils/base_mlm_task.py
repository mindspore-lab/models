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


import copy
from abc import ABC
from typing import Union, Optional, Type, Dict

import numpy as np
import mindspore
from pigmento import pnt
from mindspore import nn

from loader.task.base_batch import BertBatch
from loader.task.base_task import BaseTask
from loader.task.base_loss import TaskLoss
from loader.task.utils.base_classifiers import BertClassifier
import copy

class MLMBertBatch(BertBatch):
    def __init__(self, batch):
        super(MLMBertBatch, self).__init__(batch=batch)
        self.mask_labels = None  # type:
        self.mask_labels_col = copy.deepcopy(self.col_mask)  # type:
        self.mask_ratio = None  # type:

        self.register('mask_labels', 'mask_labels_col', 'mask_ratio')





class BaseMLMTask(BaseTask, ABC):
    name = 'base-mlm'
    mask_scheme = 'MASK_{col}'
    mask_col_ph = '{col}'
    cls_module: Union[Type[BertClassifier]]
    col_order: list
    batcher: Union[Type[MLMBertBatch]]

    def __init__(
            self,
            select_prob=0.4,
            mask_prob=0.8,
            random_prob=0.1,
            loss_pad=-100,
            apply_cols=None,
            **kwargs
    ):
        super(BaseMLMTask, self).__init__()

        self.select_prob = select_prob
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.loss_pad = loss_pad

        self.apply_cols = apply_cols
        self.loss_fct = nn.CrossEntropyLoss()

    def get_col_order(self, order):
        order = list(map(lambda x: x.name, order.order))
        if not self.apply_cols:
            return copy.deepcopy(order)
        return list(filter(lambda col: col in self.apply_cols, order))

    def get_expand_tokens(self):
        return [self.mask_scheme]

    def get_mask_token(self, col_name):
        # print('self.dataset.TOKENS',self.dataset.TOKENS)
        TOKENS = {'PAD': 0, 'BOS': 1, 'SEP': 2, 'MASK_k_cluster': 3, 'MASK_p_cluster': 4}
        return TOKENS[self.mask_scheme.replace(self.mask_col_ph, col_name)]

    def prepare_batch(self, batch: MLMBertBatch):
        batch.mask_labels = mindspore.ops.ones((batch.batch_size, 103), dtype=mindspore.int64) * self.loss_pad
        # print('k_cluster1', batch.col_mask['k_cluster'])
        # batch.mask_labels_col = copy.deepcopy(batch.col_mask)

    def do_mask(self, mask, tok, vocab_size):
        tok = int(tok)

        if np.random.uniform() < self.select_prob:
            mask_type = np.random.uniform()
            # print()
            if mask_type < 1:#self.mask_prob:
                return mask, tok, True
            elif mask_type < self.mask_prob + self.random_prob:
                return np.random.randint(vocab_size), tok, False
            return tok, tok, False
        return tok, self.loss_pad, False

    def random_mask(self, batch: MLMBertBatch, col_name):
        vocab_size = 10
        for i_batch in range(batch.batch_size):
            # if batch.batch_size==1:
            #     break
            # print('batch.batch_size',batch.batch_size)
            for i_tok in range(103):
                # print('self.dataset.max_sequence',self.dataset.max_sequence)
                # if batch.col_mask[col_name][i_batch][i_tok]:
                if batch.col_mask[col_name][i_batch][i_tok]==1:
                    input_id, mask_label, use_special_col = self.do_mask(
                        mask=self.get_mask_token(col_name),
                        tok=batch.input_ids[i_batch][i_tok],
                        vocab_size=vocab_size
                    )
                    batch.input_ids[i_batch][i_tok] = input_id
                    batch.mask_labels[i_batch][i_tok] = mask_label
                    # print('use_special_col',use_special_col)
                    if use_special_col:
                        # print('mmmm')
                        batch.col_mask[col_name][i_batch][i_tok] = 0
                        batch.col_mask['__special_id'][i_batch][i_tok] = 1
        # print('k_cluster', batch.col_mask['k_cluster'])
    def left2right_mask(self, batch: MLMBertBatch, col_name):
        for i_batch in range(batch.batch_size):
            col_start, col_end = None, None
            for i_tok in range(103):
                if batch.col_mask[col_name][i_batch][i_tok]:
                    if col_start is None:
                        col_start = i_tok
                    else:
                        col_end = i_tok
            col_end += 1

            if self.is_training:
                # print('''asdfghjkl''')
                mask_count = int((col_end - col_start) * batch.mask_ratio)
                # print('mask_count',mask_count)
                col_start = col_end - 1
            # print('col_start',col_start)
            # print('col_end',col_end)
            selected_tokens = slice(col_start, col_end)
            # print('selected_tokens',selected_tokens)
            batch.mask_labels[i_batch][selected_tokens] = batch.input_ids[i_batch][selected_tokens]
            batch.input_ids[i_batch][selected_tokens] = self.get_mask_token(col_name)
            # print('111111',batch.col_mask[col_name])
            # print('222222',batch.col_mask[col_name][i_batch][selected_tokens])
            batch.col_mask[col_name][i_batch][selected_tokens] = 0
            # print('batch.col_mask[col_name][i_batch][selected_tokens] ',batch.col_mask[col_name][i_batch][selected_tokens] )
            batch.col_mask['__special_id'][i_batch][selected_tokens] = 1

    def _init_extra_module(self):
        module_dict = dict()

        for col_name in self.col_order:
            voc = self.depot.cols[col_name].voc

            pnt(f'preparing CLS module for {col_name} - {voc.name}')
            if voc.name in module_dict:
                pnt(f'exist in module dict, skip')
                continue

            module_dict[voc.name] = self.cls_module.create(
                config=self.model_init.model_config,
                key=voc.name,
                vocab_size=voc.size,
            )
            pnt(f'created')
        return nn.CellDict(module_dict)

    def _produce_output(self, last_hidden_state, batch: Union[MLMBertBatch]):
        output_dict = dict()
        for vocab_name in self.extra_module:
            classification_module = self.extra_module[vocab_name]
            output_dict[vocab_name] = classification_module(last_hidden_state)
        return output_dict

    def calculate_loss(self, batch: MLMBertBatch, output, **kwargs) -> TaskLoss:
        weight = kwargs.get('weight', 1)

        mask_labels = batch.mask_labels  # type:

        total_loss = mindspore.Tensor(0, dtype=mindspore.float32)
        for col_name in self.col_order:
            voc = self.depot.cols[col_name].voc
            # vocab_size = self.depot.get_vocab_size(col_name)

            mask_labels_col = batch.mask_labels_col[col_name] # type:

            col_mask = batch.col_mask[col_name]
            masked_elements = mindspore.ops.not_equal(col_mask, mask_labels_col)  # type:
            if not mindspore.ops.sum(masked_elements):
                continue

            distribution = mindspore.ops.masked_select(
                output[voc.name], masked_elements.unsqueeze(dim=-1)).view(-1, voc.size)
            col_labels = mindspore.ops.masked_select(mask_labels, masked_elements)

            loss = self.loss_fct(
                distribution,
                col_labels
            )
            total_loss += loss * weight
        return TaskLoss(loss=total_loss)
