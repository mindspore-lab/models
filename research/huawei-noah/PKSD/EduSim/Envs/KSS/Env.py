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
# coding: utf-8

from copy import deepcopy
import networkx as nx
import random
from EduSim.Envs.meta import Env

import numpy as np
from EduSim.Envs.KSS.meta.Learner import LearnerGroup, Learner
from EduSim.Envs.shared.KSS_KES import episode_reward
from EduSim.spaces import ListSpace
from .meta import KSSItemBase, KSSScorer
from .utils import load_environment_parameters
from EduSim.utils import get_graph_embeddings
import copy
from gensim.models import Word2Vec

__all__ = ["KSSEnv"] # __all__，其他文件中使用from xxx import *导入该文件时，只会导入 __all__ 列出的成员，可以其他成员都被排除在外。


class KSSEnv(Env):
    def __init__(self, seed=None, initial_step=20):
        self.type = 'KSS'
        self.env_name = 'KSS'
        self.graph_embeddings = get_graph_embeddings('KSS')
        # self.word2vec_model = Word2Vec.load(f'{get_proj_path()}/EduSim/Envs/meta_data/KSSwvModel.pt')
        self.random_state = np.random.RandomState(seed)

        parameters = load_environment_parameters()
        self.knowledge_structure = parameters["knowledge_structure"]
        self._item_base = KSSItemBase(  # 获得item的信息，包括knowledge和difficulty属性
            parameters["knowledge_structure"],
            parameters["learning_order"],
            items=parameters["items"]  # 这里item加载进来是一个字典{"0": {"knowledge": 0},}
        )
        self.learning_item_base = deepcopy(self._item_base)  # 得到learning_items
        self.learning_item_base.drop_attribute()  # learning_items相对于test_items没有难度这一属性
        self.test_item_base = self._item_base  # 得到test_items
        self.scorer = KSSScorer(parameters["configuration"].get("binary_scorer", True))  # 采取0、1打分制

        self.action_space = ListSpace(self.learning_item_base.item_id_list, seed=seed)  # 获得agent的action space

        self.learners = LearnerGroup(self.knowledge_structure, seed=seed)  # learners有知识结构图和随机种子两个属性

        self._order_ratio = parameters["configuration"]["order_ratio"]  # 1.0
        self._review_times = parameters["configuration"]["review_times"]  # 1
        self._learning_order = parameters["learning_order"]
        self._topo_order = list(nx.topological_sort(self.knowledge_structure))  # 通过knowledge_structure得到知识点的拓扑排序
        self._initial_step = parameters["configuration"]["initial_step"] if initial_step is None else initial_step
        self._learner = None
        self._initial_score = None
        self._exam_reduce = "sum" if parameters["configuration"].get("exam_sum", True) else "ave"

        # self._learner = next(self.learners)  # 构建learner（learning target、state、knowledge_structure）
        # self._initial_logs(self._learner)  # 得到learner在initial_step下的学习情况
        # self.fixed_learner = copy.deepcopy(self._learner)

        # self.test_path_for_learner()

    @property
    def parameters(self) -> dict:
        return {
            "knowledge_structure": self.knowledge_structure,
            "action_space": self.action_space,
            "learning_item_base": self.learning_item_base

        }

    def _initial_logs(self, learner: Learner):
        logs = []
        if random.random() < self._order_ratio:  # 由于_order_ratio等于1，所以条件几乎必然成立
            while len(logs) < self._initial_step:
                if logs and logs[-1][1] == 1 and len(set([e[0] for e in logs[-3:]])) > 1:
                    for _ in range(self._review_times):
                        if len(logs) < self._initial_step - self._review_times:
                            learning_item_id = logs[-1][0]
                            test_item_id, score = self.learn_and_test(learner, learning_item_id)
                            logs.append([test_item_id, score])
                        else:
                            break
                    learning_item_id = logs[-1][0]
                elif logs and logs[-1][1] == 0 and random.random() < 0.7:
                    learning_item_id = logs[-1][0]
                elif random.random() < 0.9:
                    for knowledge in self._topo_order:  # knowledge_structure 拓扑排序后
                        test_item_id = self.test_item_base.knowledge2item[knowledge].id  # 按照_topo_order 作为测试项目
                        if learner.response(self.test_item_base[test_item_id]) < 0.6:  # 测试情况不理想，就进行learn_and_test的学习
                            break
                    else:  # pragma: no cover
                        break
                    learning_item_id = test_item_id
                else:
                    learning_item_id = self.random_state.choice(list(self.learning_item_base.index))
                test_item_id, score = self.learn_and_test(learner, learning_item_id)
                logs.append([test_item_id, score])  # logs是作为environment给agent的observation
        else:
            while len(logs) < self._initial_step:
                if random.random() < 0.9:
                    for knowledge in self._learning_order:
                        test_item_id = self.test_item_base.knowledge2item[knowledge].id
                        if learner.response(self.test_item_base[test_item_id]) < 0.6:
                            break
                    else:
                        break
                    learning_item_id = test_item_id
                else:
                    learning_item_id = self.random_state.choice(self.learning_item_base.index)

                item_id, score = self.learn_and_test(learner, learning_item_id)
                logs.append([item_id, score])

        learner.update_logs(logs)

    def learn_and_test(self, learner: Learner, item_id):
        learning_item = self.learning_item_base[item_id]
        test_item_id = item_id  # test_item与learner_item是同样的
        test_item = self.test_item_base[test_item_id]
        score = self.scorer(learner.response(test_item), test_item.attribute)
        learner.learn(learning_item)
        return item_id, score

    def _exam(self, learner: Learner, detailed=False, reduce=None) -> (dict, int, float):
        if reduce is None:  # 注：这里是sum，在meta_data/configuration里面
            reduce = self._exam_reduce  # 测验成果是算总和，还是平均
        knowledge_response = {}  # dict
        for test_knowledge in learner.target:
            item = self.test_item_base.knowledge2item[test_knowledge]
            knowledge_response[test_knowledge] = [item.id, self.scorer(learner.response(item), item.attribute)]
        if detailed:
            return knowledge_response
        elif reduce == "sum":
            return np.sum([v for _, v in knowledge_response.values()])  # np.sum   []:list   knowledge_response
        elif reduce in {"mean", "ave"}:
            return np.average([v for _, v in knowledge_response.values()])
        else:
            raise TypeError("unknown reduce type %s" % reduce)  # unknown reduce type

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)  # 构建learner（learning target、state、knowledge_structure）
        self._initial_logs(self._learner)
        self._initial_score = self._exam(self._learner)
        while self._initial_score == len(self._learner.target):
            self._learner = next(self.learners)  # 构建learner（learning target、state、knowledge_structure）
            self._initial_logs(self._learner)  # 得到learner在initial_step下的学习情况
            self._initial_score = self._exam(self._learner)  # 测试learner初始的学习成绩
        # self._learner = copy.deepcopy(self.fixed_learner)
        # # # dif targets
        # # self._learner._target = set(self.random_state.choice(len(self.knowledge_structure), self.random_state.randint(3, len(self.knowledge_structure))))
        # # self._learner.learning_model._target = self._learner._target
        # # one targets
        # # self._learner._target = set(list([2, 3, 5, 8]))
        # # self._learner.learning_model._target = self._learner._target
        #
        # self._initial_score = self._exam(self._learner)  # 测试learner初始的学习成绩
        return self._learner.profile, self._exam(self._learner, detailed=True)  # learner的profile包含id、logs、target

    def end_episode(self, *args, **kwargs):
        observation = self._exam(self._learner, detailed=True)  # 一个episode结束后测试learner的掌握情况，得到所有信息
        initial_score, self._initial_score = self._initial_score, None
        final_score = self._exam(self._learner)  # 只要一个总分的分数
        reward = episode_reward(initial_score, final_score, len(self._learner.target))  # 一个episode-reward的计算方法
        done = final_score == len(self._learner.target)  # 因为是binary-score，如果总分与learning_target个数一致，则说明学习完成
        info = {"initial_score": initial_score, "final_score": final_score}

        return observation, reward, done, info

    def step(self, learning_item_id, *args, **kwargs):  # point-wise
        a = self._exam(self._learner)  # 测试learner对于learning_target中的knowledge掌握情况
        observation = self.learn_and_test(self._learner, learning_item_id)  # 学习agent选择的learning_item_id
        self._learner._logs.append([observation[0], observation[1]])
        b = self._exam(self._learner)  # 重新测试learner的掌握情况
        # print(self._learner.state)
        return observation, b - a, b == len(self._learner.target), None  # 两者之差作为reward

    def n_step(self, learning_path, *args, **kwargs):  # sequence-wise
        exercise_history = []
        a = self._exam(self._learner)
        for learning_item_id in learning_path:
            item_id, score = self.learn_and_test(self._learner, learning_item_id)
            exercise_history.append([item_id, score])
        b = self._exam(self._learner)
        return exercise_history, b - a, b == len(self._learner.target), None

    def reset(self):
        self._learner = None

    def render(self, mode='human'):
        if mode == "log":
            return "target: %s, state: %s" % (
                self._learner.target, dict(self._exam(self._learner))
            )

    def test_path_for_learner(self):
        end_flag = False
        inc = 0
        targets_list = [0, 3, 7]
        while not end_flag:
            # generate learner
            knowledge = self.knowledge_structure.nodes
            self._learner = Learner(
                [min(-3+0.1*inc, 0) - (0.1 * i) for i, _ in enumerate(knowledge)],  # 随机初始化 learner state
                self.knowledge_structure,
                set(targets_list),
                # 随机初始化learning target,经测试基本为3-6个
            )  # 构建learner（learning target、state、knowledge_structure）
            self._initial_logs(self._learner)
            self._initial_score = self._exam(self._learner)
            while self._initial_score == len(self._learner.target):
                self._learner = next(self.learners)  # 构建learner（learning target、state、knowledge_structure）
                self._initial_logs(self._learner)  # 得到learner在initial_step下的学习情况
                self._initial_score = self._exam(self._learner)  # 测试learner初始的学习成绩

            # fix this learner
            self.fixed_learner = copy.deepcopy(self._learner)

            # given paths
            path_lists = []
            path_lists.append([2, 0, 0, 0, 0, 0, 5, 2, 1, 0, 5, 2, 2, 4, 0, 5, 2, 7, 2, 9])  # ac
            path_lists.append([7, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 0, 0, 0, 0, 0, 0, 0, 2]) # cseal

            # run
            path_with_score_promotion = [[], [], []]
            path_with_state_promotion = [[], [], []]
            for i, path in enumerate(path_lists):
                self._learner = copy.deepcopy(self.fixed_learner)
                self._initial_score = self._exam(self._learner)
                initial_state = np.array([self._learner._state[target] for target in targets_list])
                print(f'initial_score:{self._initial_score}')

                for item_id in path:
                    observation, promotion,done, _ = self.step(str(item_id))
                    current_score = self._exam(self._learner)
                    score_promotion = episode_reward(self._initial_score, current_score, len(self._learner.target))
                    path_with_score_promotion[i].append([item_id, score_promotion])

                    current_state = np.array([self._learner._state[target] for target in targets_list])
                    state_promotion = np.mean(((current_state - initial_state) + 4) / 9)
                    path_with_state_promotion[i].append([item_id, current_score])
                print(path_with_state_promotion[i])
            if path_with_score_promotion[-1][-1][1] == 1.0:
                end_flag = True
            inc += 0.1
        assert 0