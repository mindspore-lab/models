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
import os
import json
import time
import networkx as nx
import numpy as np
import mindspore
from EduSim.Envs.KES.meta.Learner import LearnerGroup, Learner
from EduSim.Envs.shared.KSS_KES import episode_reward
from EduSim.spaces import ListSpace
from EduSim.deep_model import DKTnet
from EduSim.Envs.meta import Item
from EduSim.Envs.meta import Env
from EduSim.utils import get_proj_path, get_graph_embeddings
from .meta import KESScorer

__all__ = ["KESEnv"]


class KESEnv(Env):
    def __init__(self, dataRec_path, seed=None):
        super(KESEnv, self).__init__()
        self.random_state = np.random.RandomState(seed)
        self.graph_embeddings = get_graph_embeddings('KES_junyi')
        self.dataRec_path = dataRec_path

        # 得到知识图的num_skills
        with open(f'{get_proj_path()}/data/dataProcess/junyi/graph_vertex.json') as f:
            # exercise-id的对应字典，存储 {exercise_str:id},length即为835
            ku_dict = json.load(f)

        # 获得知识图--junyi数据集自带的
        with open(f'{get_proj_path()}/data/dataProcess/junyi/prerequisite.json') as f:
            prerequisite_edges = json.load(f)
            self.knowledge_structure = nx.DiGraph()
            # add this line by lqy
            self.knowledge_structure.add_nodes_from(ku_dict.values())
            self.knowledge_structure.add_edges_from(prerequisite_edges)
            self._topo_order = list(nx.topological_sort(self.knowledge_structure))
            assert not list(nx.algorithms.simple_cycles(self.knowledge_structure)), "loop in DiGraph"

        # 获取知识图--自己构建的transition graph
        # self.knowledge_structure = nx.DiGraph()
        # with open(f"{get_proj_path()}/data/dataProcess/junyi/nxgraph.pkl", "rb") as file:
        #     self.knowledge_structure = pickle.loads(file.read())
        # # 打印topo_order
        # self._topo_order = list(nx.topological_sort(self.knowledge_structure))
        # # print(self._topo_order)
        # assert not list(nx.algorithms.simple_cycles(self.knowledge_structure)), "loop in DiGraph"

        # 设置item_base
        self.num_skills = len(ku_dict)
        self.max_sequence_length = 300
        self.feature_dim = 2 * self.num_skills
        self.embed_dim = 600  # from paper
        self.hidden_size = 900  # from paper
        self.item_list = [i for i in range(len(ku_dict))]
        self.learning_item_base = [Item(item_id=i, knowledge=i) for i in self.item_list]

        # DKTnet
        dkt_input_dict = {
            'input_size': self.feature_dim,
            'emb_dim': self.embed_dim,
            'hidden_size': self.hidden_size,
            'num_skills': self.num_skills,
            'nlayers': 2,
            'dropout': 0.0,
        }
        self.DKTnet = DKTnet(dkt_input_dict)
        directory = f'{get_proj_path()}/EduSim/Envs/KES/meta_data'
        dkt_file = f'{directory}/env_weights/ValBest.ckpt'
        # dkt_file = f'{directory}/env_weights/SelectedValBest.ckpt'
        if os.path.exists(dkt_file):
            param_dict = mindspore.load_checkpoint(dkt_file)
            _, _ = mindspore.load_param_into_net(self.DKTnet, param_dict)
        self.scorer = KESScorer()
        self.action_space = ListSpace(self.item_list, seed=seed)  # 获得agent的action space

        # learners
        self.learners = LearnerGroup(self.dataRec_path, seed=seed)  # learners有知识结构图和随机种子两个属性
        self._learner = None
        self._initial_score = None
        self.episode_start_time = time.time()
        self.episode_end_time = time.time()
        self.type = 'KES_junyi'
        self.env_name = 'KESjunyi'

    @property
    def parameters(self) -> dict:
        return {
            "action_space": self.action_space
        }

    def learn_and_test(self, learner: Learner, item_id):
        state = learner.state
        score = self.scorer.response_function(state, item_id)
        learner.learn(item_id, score)  # 添加学习记录
        self.update_learner_state()
        return item_id, score

    def _exam(self, learner: Learner, detailed=False, reduce="sum") -> (dict, int, float):
        state = learner.state
        knowledge_response = {}  # dict
        for test_item in learner.target:
            knowledge_response[test_item] = [test_item, self.scorer.response_function(state, test_item)]
        if detailed:
            return_thing = knowledge_response
        elif reduce == "sum":
            return_thing = np.sum([v for _, v in knowledge_response.values()])  # np.sum   []:list   knowledge_response
        elif reduce in {"mean", "ave"}:
            return_thing = np.average([v for _, v in knowledge_response.values()])
        else:
            raise TypeError("unknown reduce type %s" % reduce)  # unknown reduce type
        return return_thing

    def update_learner_state(self):
        logs = self._learner.profile['logs']
        sequence_length = len(logs)
        input_data = self.get_feature_matrix(logs).unsqueeze(0)  # [bz,sequence_length,feture_dim]
        self._learner._state = mindspore.ops.sigmoid(
            self.DKTnet(input_data).permute(1, 0, 2).squeeze(0)[sequence_length - 1])

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)  # 构建learner（learning target、state、knowledge_structure）
        self.update_learner_state()
        self._initial_score = self._exam(self._learner)  # 测试learner初始的学习成绩
        while self._initial_score >= len(self._learner.target):
            self._learner = next(self.learners)  # 构建learner（learning target、state、knowledge_structure）
            self.update_learner_state()
            self._initial_score = self._exam(self._learner)  # 测试learner初始的学习成绩
        return self._learner.profile, self._exam(self._learner, detailed=True)  # learner的profile包含id、logs、target

    def end_episode(self, *args, **kwargs):
        observation = self._exam(self._learner, detailed=True)
        initial_score, self._initial_score = self._initial_score, None
        final_score = self._exam(self._learner)  # 只要一个总分的分数
        reward = episode_reward(initial_score, final_score, len(self._learner.target))
        done = final_score == len(self._learner.target)  # 因为是binary-score，如果总分与learning_target个数一致，则说明学习完成
        info = {"initial_score": initial_score, "final_score": final_score}
        self.episode_end_time = time.time()
        # print('episode_env_time:' + str(self.episode_end_time - self.episode_start_time))
        return observation, reward, done, info

    def step(self, learning_item_id, *args, **kwargs):  # point-wise
        a = self._exam(self._learner)  # 测试learner对于learning_target中的knowledge掌握情况
        observation = self.learn_and_test(self._learner, learning_item_id)
        b = self._exam(self._learner)  # 重新测试learner的掌握情况
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
            return_thing = "target: %s, state: %s" % (
                self._learner.target, dict(self._exam(self._learner))
            )
        else:
            return_thing = 'for else return'
        return return_thing

    def get_feature_matrix(self, session):
        input_data = np.zeros(shape=(max(1, len(session)), self.feature_dim), dtype=np.float32)
        # 对输入x进行编码
        j = 0
        while j < len(session):
            problem_id = session[j][0]
            if session[j][1] == 0:  # 对应问题回答错误
                input_data[j][problem_id] = 1.0
            elif session[j][1] == 1:  # 对应问题回答正确
                input_data[j][problem_id + self.num_skills] = 1.0
            j += 1
        return mindspore.Tensor(input_data)
