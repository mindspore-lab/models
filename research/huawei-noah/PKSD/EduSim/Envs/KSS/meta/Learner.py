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

import numpy as np
import random
import math

import networkx as nx
from EduSim.Envs.meta import MetaLearner, MetaInfinityLearnerGroup, MetaLearningModel, Item
from EduSim.Envs.shared.KSS_KES.KS import influence_control

__all__ = ["Learner", "LearnerGroup"]


class LearningModel(MetaLearningModel):
    def __init__(self, state, learning_target, knowledge_structure, last_visit=None):
        self._state = state
        self._target = learning_target
        self._ks = knowledge_structure
        self._ks_last_visit = last_visit

    def step(self, state, knowledge):
        too_away_penalty = 0.8
        if self._ks_last_visit is not None:
            if knowledge not in influence_control(
                    self._ks, state, self._ks_last_visit, allow_shortcut=False, target=self._target,
            )[0]:  # 根据knowledge_structure 挑出 下一步可以学习的candidates
                too_away_penalty = 0.1
        self._ks_last_visit = knowledge  # 如果knowledge在candidates中，则学习这个知识点

        # capacity growth function
        # 5是规定掌握程度的最大值,如果你的前继离5越远，那你学习这个ks的discount就越大，即增长越少
        discount = math.exp(sum([(5 - state[node]) for node in self._ks.predecessors(knowledge)] + [0]))
        ratio = 1 / discount
        inc = (5 - state[knowledge]) * ratio * 0.5 * too_away_penalty

        def _promote(_ind, _inc):
            state[_ind] += _inc
            if state[_ind] > 5:  # 保证state最高值为5
                state[_ind] = 5
            for node in self._ks.successors(_ind):  # 掌握了当前知识点，对于它后续的知识点的掌握情况都会有所提升
                _promote(node, _inc * 0.5)

        _promote(knowledge, inc)  # 更新learner对各个knowledge的掌握情况，即state


class Learner(MetaLearner):
    def __init__(self,
                 initial_state,
                 knowledge_structure: nx.DiGraph,
                 learning_target: set,
                 _id=None,
                 seed=None):
        super(Learner, self).__init__(user_id=_id)
        learning_target = [int(i) for i in learning_target]
        self.learning_model = LearningModel(
            initial_state,
            learning_target,
            knowledge_structure,
        )  # 初始化learner的学习模型

        # 初始化learner的基本信息：state/target/knowledge_structure/logs
        self.structure = knowledge_structure
        self._state = initial_state  # state表示learner对每个knowledge的掌握情况
        self._target = learning_target
        self._logs = []
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

    def learn(self, learning_item: Item):
        self.learning_model.step(self._state, learning_item.knowledge)

    @property
    def state(self):
        return self._state

    def response(self, test_item: Item) -> ...:
        return self._state[test_item.knowledge]

    @property
    def target(self):
        return self._target


class LearnerGroup(MetaInfinityLearnerGroup):
    def __init__(self, knowledge_structure, seed=None):
        super(LearnerGroup, self).__init__()
        self.knowledge_structure = knowledge_structure
        self.random_state = np.random.RandomState(seed)

    def __next__(self):
        knowledge = self.knowledge_structure.nodes
        return Learner(
            [self.random_state.randint(-3, 0) - (0.1 * i) for i, _ in enumerate(knowledge)],  # 随机初始化 learner state
            self.knowledge_structure,
            set(self.random_state.choice(len(knowledge), self.random_state.randint(3, len(knowledge)))),
            # 随机初始化learning target,经测试基本为3-6个
        )
