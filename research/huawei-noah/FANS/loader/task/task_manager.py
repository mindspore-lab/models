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


from oba import Obj
from pigmento import pnt
from loader.task.bert.curriculum_cluster_mlm_task import CurriculumClusterMLMTask



class TaskManager:
    TASKS_LIST = [
        CurriculumClusterMLMTask,
    ]

    TASKS = {task.name: task for task in TASKS_LIST}
    print('TASKS',TASKS)
    def __init__(self, project_exp):
        self.tasks = []
        self.applied_task_indexes = []
        self.injection_task = None

        for task_config in project_exp.tasks:
            if task_config.name not in TaskManager.TASKS:
                raise ValueError(f'No matched task: {task_config.name}')
            print('task_config.name',task_config.name)
            task_class = TaskManager.TASKS[task_config.name]
            params = Obj.raw(task_config.params)

            task = task_class(**params)
            self.tasks.append(task)
            if not task_config.only_initialization:
                self.applied_task_indexes.append(len(self.tasks) - 1)
            if task.injection:
                self.injection_task = task

            pnt(task_config.name, 'params:', params)

        self.expand_tokens = []
        for task in self.tasks:
            self.expand_tokens.extend(task.get_expand_tokens())
        pnt('expand tokens:', self.expand_tokens)
