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
import json
import copy
import os
import numpy as np
from EduSim.Envs.meta import MetaLearner, MetaInfinityLearnerGroup

__all__ = ["Learner", "LearnerGroup"]


class Learner(MetaLearner):
    def __init__(self,
                 initial_log,
                 learning_target: set,
                 _id=None,
                 seed=None):
        super(Learner, self).__init__(user_id=_id)

        # 初始化learner的基本信息：state/target/knowledge_structure/logs
        self._target = learning_target
        self._logs = initial_log
        self._state = []
        self.random_state = np.random.RandomState(seed)

    def update_logs(self, logs):
        self._logs = logs

    @property
    def profile(self):
        return {
            "id": self.id,
            "logs": self._logs,
            "target": self.target
        }

    def learn(self, learning_item, score):
        self._logs.append([learning_item, score])

    @property
    def state(self):
        return self._state

    def response(self, test_item) -> ...:
        return self._state[test_item]

    @property
    def target(self):
        return self._target


class LearnerGroup(MetaInfinityLearnerGroup):
    def __init__(self, dataRec_path, seed=None):
        super(LearnerGroup, self).__init__()
        self.data_path = dataRec_path
        self.random_state = np.random.RandomState(seed)
        if not os.path.isdir(self.data_path) and not 'npz' in self.data_path:
            with open(self.data_path, 'r', encoding="utf-8") as f:
                self.datatxt = f.readlines()

    def __next__(self):
        session = [[0, 0]]
        learning_targets = set()
        while len({log[0] for log in session}) < 20:
            index = self.random_state.randint(len(self.datatxt))
            session = json.loads(self.datatxt[index])
            learning_targets = {step[0] for i, step in enumerate(session) if i >= 0.8 * len(session)}
            # learning_target = set(self.random_state.choice(835, self.random_state.randint(3, 10)))

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
