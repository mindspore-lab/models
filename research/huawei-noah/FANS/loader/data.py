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


from pigmento import pnt


class TaskInitializer:
    def __init__(self,  model_init, device):
        self.bert_init = model_init
        self.device = device

        self.depot = dict()

    def register(self, *tasks):
        for task in tasks:
            self.depot[task.name] = task
            task.init(
                model_init=self.bert_init,
                device=self.device,
            )
        return self

    def __getitem__(self, item):
        return self.depot[item]

    def get_extra_modules(self):
        extra_modules = dict()
        pnt('create extra modules')
        print('self.depot',self.depot['cu-cluster-mlm'])
        # self.depot = {'cu-cluster-mlm': cu-cluster-mlm}
        for task_name in self.depot:
            print(task_name)
            extra_module = self.depot[task_name].init_extra_module()
            extra_modules[task_name] = extra_module
        return extra_modules


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
