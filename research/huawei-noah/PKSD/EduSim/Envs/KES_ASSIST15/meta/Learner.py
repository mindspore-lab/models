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
# -*- coding:utf-8 _*-
import os
import copy
import numpy as np
from EduSim.Envs.KES.meta.Learner import LearnerGroup as KESLearnerGroup
from EduSim.Envs.KES.meta.Learner import Learner


class KESASSIST15LeanerGroup(KESLearnerGroup):
    def __init__(self, dataRec_path, seed=None):
        super().__init__(dataRec_path, seed)
        self.data_path = dataRec_path
        self.random_state = np.random.RandomState(seed)

    def __next__(self):
        dataset_number = self.random_state.randint(1, 6)
        all_students = os.listdir(f'{self.data_path}{dataset_number}/train/')
        session = [[0, 0]]
        learning_targets = set()
        while len({log[0] for log in session}) <= 6 or len(learning_targets) <= 1:
            index = self.random_state.randint(len(all_students))
            with open(f'{self.data_path}{dataset_number}/train/{index}.csv', 'r') as f:
                data = f.readlines()[1:]
            session = [[int(line.rstrip().split(',')[0]) - 1,
                        int(line.rstrip().split(',')[1])] for i, line in enumerate(data)]
            learning_targets = {step[0] for i, step in enumerate(session) if i >= 0.8 * len(session)}
            # learning_target = set(self.random_state.choice(100, self.random_state.randint(3, 10)))
        # if len(learning_targets) > 20:
        #     targets_num = random.randint(2, 20)
        #     learning_targets = random.sample(learning_targets, targets_num)

        initial_log = copy.deepcopy(session[:int(len(session) * 0.6)])
        # initial_max_session_length = 20
        # if len(initial_log) > initial_max_session_length:
        #     initial_log = initial_log[:initial_max_session_length]
        return Learner(
            initial_log=initial_log,
            learning_target=learning_targets,
        )
