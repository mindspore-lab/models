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

from tinybert import BertConfig
from loader.init.model_init import ModelInit


class BertInit(ModelInit):
    def __init__(
            self,
            num_hidden_layers=12,
            num_attention_heads=12,
            **kwargs
    ):
        super(BertInit, self).__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

    def load_model_config(self):
        return BertConfig(
            vocab_size=1,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.hidden_size * 4,
            max_position_embeddings=103,
            type_vocab_size=2,
        )
