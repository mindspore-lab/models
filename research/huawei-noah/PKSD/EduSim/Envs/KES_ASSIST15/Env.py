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
import pickle
import time
import os
import networkx as nx
import numpy as np
import mindspore

from EduSim.Envs.KES_ASSIST15.meta.Learner import KESASSIST15LeanerGroup
from EduSim.spaces import ListSpace
from EduSim.deep_model import DKTnet
from EduSim.Envs.meta import Item
from EduSim.Envs.KES import KESEnv
from EduSim.utils import get_proj_path, get_graph_embeddings

from .meta import KESASSISTScorer
__all__ = ["KESASSISTEnv"]


class KESASSISTEnv(KESEnv):
    def __init__(self, dataRec_path, seed=None):
        super().__init__(dataRec_path, seed)
        self.type = 'KES'
        self.env_name = 'KESassist15'
        self.random_state = np.random.RandomState(seed)
        self.graph_embeddings = get_graph_embeddings('KES_ASSIST15')
        self.dataRec_path = dataRec_path

        # 获得知识图
        self.knowledge_structure = nx.DiGraph()
        with open(f"{get_proj_path()}/data/dataProcess/ASSISTments2015/nxgraph.pkl", "rb") as file:
            self.knowledge_structure = pickle.loads(file.read())
        # 打印topo_order
        self._topo_order = list(nx.topological_sort(self.knowledge_structure))
        # print(self._topo_order)
        assert not list(nx.algorithms.simple_cycles(self.knowledge_structure)), "loop in DiGraph"

        # 设置item_base
        self.num_skills = 100
        self.max_sequence_length = 300
        self.feature_dim = 2 * self.num_skills
        self.embed_dim = 64  # from paper
        self.hidden_size = 128  # from paper
        self.item_list = [i for i in range(self.num_skills)]
        self.learning_item_base = [Item(item_id=i, knowledge=i) for i in self.item_list]

        # DKTnet
        dkt_para_dict = {
            'input_size': self.feature_dim,
            'emb_dim': self.embed_dim,
            'hidden_size': self.hidden_size,
            'num_skills': self.num_skills,
            'nlayers': 2,
            'dropout': 0.0,
        }
        self.DKTnet = DKTnet(dkt_para_dict)
        directory = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST15/meta_data'
        dkt_file = f'{directory}/env_weights/ValBest.ckpt'
        # dkt_file = f'{directory}/env_weights/SelectedValBest.ckpt'
        if os.path.exists(dkt_file):
            param_dict = mindspore.load_checkpoint(dkt_file)
            _, _ = mindspore.load_param_into_net(self.DKTnet, param_dict)
        else:
            raise ValueError('dkt net not trained yet!')
        self.scorer = KESASSISTScorer()
        self.action_space = ListSpace(self.item_list, seed=seed)  # 获得agent的action space

        # learners
        self.learners = KESASSIST15LeanerGroup(self.dataRec_path, seed=seed)  # learners有知识结构图和随机种子两个属性
        self._learner = None
        self._initial_score = None
        self.episode_start_time = time.time()
        self.episode_end_time = time.time()
