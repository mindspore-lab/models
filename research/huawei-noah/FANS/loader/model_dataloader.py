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
from mindspore.dataset import GeneratorDataset



class ModelDataLoader(GeneratorDataset):
    def __init__(self, dataset, tasks, column_names,**kwargs):
        super().__init__(
            source=dataset,
            column_names=column_names,
            # **kwargs
        )

        self.auto_dataset = dataset
        self.tasks = tasks

    def start_epoch(self, current_epoch, total_epoch):
        for task in self.tasks:
            task.start_epoch(current_epoch, total_epoch)
        return self

    def test(self):
        for task in self.tasks:
            task.test()
        return self

    def eval(self):
        for task in self.tasks:
            task.eval()
        return self

    def train(self):
        for task in self.tasks:
            task.train()
        return self

    def __iter__(self):
        iterator = super().__iter__()

        while True:
            try:
                batch = next(iterator)
                task = random.choice(self.tasks)
                batch = task.rebuild_batch(batch)  # t
                batch.task = task
                yield batch
            except StopIteration:
                return
