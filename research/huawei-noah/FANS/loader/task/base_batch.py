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

class BaseBatch:
    def __init__(self, batch):
        self.append_info = batch['index']  # type:
        self.task = None  # type:
        self._registered_items = set()
        self.register('append_info', 'task')

    def register(self, *keys):
        self._registered_items.update(set(keys))

    def export(self):
        batch = dict()
        for key in self._registered_items:
            value = getattr(self, key)
            if isinstance(value, BaseBatch):
                value = value.export()
            batch[key] = value
        return batch


class BertBatch(BaseBatch):
    def __init__(self, batch):
        super().__init__(batch=batch)
        self.input_ids = batch['input_ids']  # type:
        self.attention_mask = batch['attention_mask']  # type
        self.segment_ids = batch['segment_ids']  # type:
        self.col_mask = {'k_cluster':batch['k_cluster'],
                         'p_cluster': batch['p_cluster'],
                         '__special_id': batch['__special_id']}  # type:
        self.attr_ids = {'k_cluster': {'k_global': batch['k_global'],'k_local': batch['k_local']},
                         'p_cluster': {'p_global': batch['p_global'],'p_local': batch['p_local']}}
        self.batch_size = int(self.input_ids.shape[0])
        self.seq_size = int(self.input_ids.shape[1])
        # print('111',self.batch_size)
        # print('222',self.seq_size)
        self.register('input_ids', 'attention_mask', 'segment_ids', 'col_mask', 'attr_ids')



