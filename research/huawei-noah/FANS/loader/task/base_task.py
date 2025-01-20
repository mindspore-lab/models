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


from typing import Dict, Optional, Union, Type

import mindspore
from mindspore import nn

from loader.init.model_init import ModelInit
from loader.task.base_batch import BertBatch, BaseBatch
from loader.task.base_loss import TaskLoss


class BaseTask:
    name: str
    injection = None
    batcher: Union[Type[BertBatch]]

    def __init__(self):
        self.dataset = None  # type:
        self.depot = None  # type:
        self.model_init = None  # type:
        self.device = None

        self.extra_module = None  # type:
        self.is_training = True
        self.is_validating = False
        self.is_testing = False

        self.injection_modes = []

    """
    Display
    """

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    """
    Task mode
    """

    def test(self):
        self.is_testing = True
        self.is_training = self.is_validating = False

    def eval(self):
        self.is_validating = True
        self.is_training = self.is_testing = False

    def train(self):
        self.is_training = True
        self.is_validating = self.is_testing = False
        # print('self.is_training',self.is_training)

    def start_epoch(self, current_epoch, total_epoch):
        return

    """
    Init
    """

    @staticmethod
    def get_expand_tokens():
        return []

    def init(self, model_init: ModelInit, device):
    # def init(self, model_init: ModelInit, device):
        self.model_init = model_init
        self.device = device

    """
    Extra module
    """

    def init_extra_module(self):
        self.extra_module = self._init_extra_module()
        return self.extra_module

    def _init_extra_module(self):
        raise NotImplementedError

    def _rebuild_batch(self, batch):
        raise NotImplementedError

    def rebuild_batch(self, batch):
        batch = self.batcher(batch)
        batch = self._rebuild_batch(batch)
        return batch

    """
    Inject dataset
    """

    # noinspection PyMethodMayBeStatic
    def sample_injector(self, sample):
        return sample

    def _injector_init(self, dataset):
        pass

    # noinspection PyMethodMayBeStatic
    def injector_init(self, dataset):
        if self.injection and dataset.mode in self.injection:
            self._injector_init(dataset)

    """
    Embedding
    """

    def get_embedding(
        self,
        batch,
        table_dict: Dict[str, nn.Embedding],
        embedding_size: int,
        enable_attrs: Union[bool, set] = True,
    ):
        input_ids = batch.input_ids  # type:
        input_embeds = mindspore.ops.zeros((*input_ids.shape, embedding_size), dtype=mindspore.float32)

        if isinstance(enable_attrs, bool):
            applied_attrs = set()
            if enable_attrs:
                for col_name in batch.col_mask:
                    if col_name in batch.attr_ids and enable_attrs:
                        for attr_name in batch.attr_ids[col_name]:
                            applied_attrs.add(attr_name)
        else:
            enable_attrs, applied_attrs = True, enable_attrs

        for col_name in batch.col_mask:
            col_mask = batch.col_mask[col_name]  # type:
            col_mask_ = col_mask.unsqueeze(-1).float()
            vocab = col_name if col_name == '__special_id' else 'cluster_id'
            atom_items = [(input_ids, vocab)]  # [B, S]
            get_vocab = {'k_global':'global_id','k_local':'local_id','p_local':'local_id','p_global':'global_id'}
            if col_name in batch.attr_ids and enable_attrs:
                for attr_name in batch.attr_ids[col_name]:
                    # print('attr_name:',attr_name)
                    # print('self.depot.get_vocab(attr_name)',self.depot.get_vocab(attr_name))
                    if attr_name in applied_attrs:
                        atom_items.append((
                            batch.attr_ids[col_name][attr_name],
                            get_vocab[attr_name]
                        ))

            for atom_inputs, atom_vocab in atom_items:
                matrix = mindspore.ops.mul(atom_inputs, col_mask)  # [B, S]
                table = table_dict[atom_vocab]
                embedding = table(matrix)
                input_embeds += mindspore.ops.mul(col_mask_, embedding)
        return input_embeds

    def produce_output(self, model_output, batch: BaseBatch):
        raise NotImplementedError

    def calculate_loss(self, batch, output, **kwargs) -> TaskLoss:
        raise NotImplementedError

    def t(self, channel, *args, **kwargs):
        return self.__getattr__(f'test__{channel}')(*args, **kwargs)

    def __getattr__(self, item: str):
        if item.startswith('test__'):
            return object.__getattribute__(self, item)

        raise ValueError(f'{self.__class__.__name__} has no {item} attribute')
