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
from pigmento import pnt
from mindspore import nn


class EmbeddingInit:
    def __init__(self):
        self.embedding_dict = dict()

    def append(self, vocab_name, vocab_type, path, freeze, global_freeze=False):
        print(vocab_name, freeze, global_freeze)
        self.embedding_dict[vocab_name] = dict(
            vocab_name=vocab_name,
            vocab_type=vocab_type,
            path=path,
            embedding=None,
            freeze=freeze or global_freeze,
        )
        return self

    def get_embedding(self, vocab_name):
        if vocab_name not in self.embedding_dict:
            return None

        embedding_info = self.embedding_dict[vocab_name]
        if embedding_info['embedding'] is not None:
            return embedding_info['embedding']

        if hasattr(self, 'get_{}_embedding'.format(embedding_info['vocab_type'])):
            getter = getattr(self, 'get_{}_embedding'.format(embedding_info['vocab_type']))
            embedding_info['embedding'] = getter(embedding_info['path'])
            return embedding_info['embedding']

class TransformEmbedding(nn.Cell):
    def __init__(self, embedding_table, from_dim, to_dim):
        super(TransformEmbedding, self).__init__()
        self.embedding_table = embedding_table
        self.linear = nn.Dense(from_dim, to_dim)

    def construct(self, indexes):
        return self.linear(self.embedding_table(indexes))


class ModelInit:
    def __init__(
            self,
            hidden_size: int = 768,
            embedding_init: EmbeddingInit = None,
            global_freeze: bool = False,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.embedding_init = embedding_init
        self.global_freeze = global_freeze

        self._embedding_tables = None
        self._model_config = None

    def load_model_config(self):
        raise NotImplementedError

    @property
    def model_config(self):
        if self._model_config:
            return self._model_config
        self._model_config = self.load_model_config()
        return self._model_config

    def get_embedding_tables(self):
        if self._embedding_tables:
            return self._embedding_tables

        embedding_tables = dict()
        required_vocabs =('cluster_id','global_id','local_id','cluster_id','global_id','local_id')

        for vocab in required_vocabs:
            embedding = self.embedding_init.get_embedding(vocab)  # type:
            if embedding is not None:
                embedding_tables[vocab] = nn.Embedding.from_pretrained(embedding)
                embedding_tables[vocab].weight.requires_grad = not self.embedding_init.is_freezing(vocab)

                if int(embedding.shape[1]) != self.hidden_size:
                    pnt.ALIGN_M_('transform embedding from', int(embedding.shape[1]), 'to', self.hidden_size)
                    embedding_tables[vocab] = TransformEmbedding(
                        embedding_table=embedding_tables[vocab],
                        from_dim=int(embedding.shape[1]),
                        to_dim=self.hidden_size
                    )

            else:
                # pnt.CREATE_M_(vocab, '( require_grad =', not self.global_freeze, '), embedding with shape', self.depot.get_vocab_size(vocab, as_vocab=True), 'x', self.hidden_size)
                table_size = {'cluster_id':10,'global_id':6264,'local_id':922}
                embedding_tables[vocab] = nn.Embedding(
                    vocab_size =table_size[vocab],
                    embedding_size =self.hidden_size
                )
                # embedding_tables[vocab].weight.requires_grad = not self.global_freeze

        # pnt.CREATE_M_(self.dataset.special_id, 'embedding with shape', len(self.dataset.special_tokens), 'x', self.hidden_size)
        embedding_tables['__special_id'] = nn.Embedding(
            vocab_size=5,
            embedding_size=self.hidden_size
        )
        # embedding_tables[self.dataset.special_id].weight.requires_grad = not self.global_freeze

        self._embedding_tables = mindspore.nn.CellDict(embedding_tables)
        return self._embedding_tables
