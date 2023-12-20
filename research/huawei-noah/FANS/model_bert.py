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


import time
import mindspore.nn as nn
from tinybert import BertModel

class AutoBert(nn.Cell):
    def __init__(self,model_init,
            device,
            task_initializer,model_class=BertModel, **kwargs):
        super(AutoBert, self).__init__()
        # super(AutoBert, self).__init__(model_class=BertModel, **kwargs)
        self.model_init = model_init
        self.device = device
        self.task_initializer = task_initializer

        self.hidden_size = self.model_init.hidden_size

        self.model = model_class(self.model_init.model_config,is_training=True)  # use compatible code
        self.embedding_tables = self.model_init.get_embedding_tables()
        self.extra_modules = self.task_initializer.get_extra_modules()
        self.timer = None

    def construct(self, batch, task,test=False):
        # task.train()
        # batch = task.rebuild_batch(batch)
        attention_mask = batch.attention_mask  # type:  # [B, S]
        segment_ids = batch.segment_ids  # type:  # [B, S]

        if isinstance(task, str):
            task = self.task_initializer[task]

        input_embeds = task.get_embedding(
            batch=batch,
            table_dict=self.embedding_tables,
            embedding_size=self.hidden_size,
        )

        start_ = time.time()
        sequence_output, pooled_output, encoder_outputs, attention_outputs = self.model(
            input_ids=input_embeds,
            input_mask=attention_mask,
            token_type_ids=segment_ids
        )  # type:
        end_ = time.time()
        if self.timer:
            self.timer.append('model', end_ - start_)
        output = task.produce_output(sequence_output, batch=batch,test=test)

        return output
