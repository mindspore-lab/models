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
from EduSim.Envs.meta import ItemBase
from networkx import Graph, DiGraph

__all__ = ["KSSItemBase"]


class KSSItemBase(ItemBase):  # 随机生成一堆difficulty的值，对应每一个knowledge赋一个difficulty，对于每一个item根据他对应的knowledge加上难度属性
    def __init__(self, knowledge_structure: (Graph, DiGraph), learning_order=None, items=None, seed=None,
                 reset_attributes=True):
        self.random_state = np.random.RandomState(seed)
        if items is None or reset_attributes:
            assert learning_order is not None
            _difficulties = list(
                sorted([self.random_state.randint(0, 5) for _ in range(len(knowledge_structure.nodes))])
            )
            difficulties = {}
            for i, node in enumerate(knowledge_structure.nodes):  # enumerate 会生成dictionary{0:'',1:''}
                difficulties[node] = _difficulties[i]

            if items is None:
                items = [
                    {
                        "knowledge": node,
                        "attribute": {
                            "difficulty": difficulties[node]
                        }
                    } for node in knowledge_structure.nodes
                ]
            elif isinstance(items, list):
                for item in items:
                    item["attribute"] = {"difficulty": difficulties[item["knowledge"]]}
            elif isinstance(items, dict):
                for item in items.values():
                    item["attribute"] = {"difficulty": difficulties[item["knowledge"]]}  # 直接改变items本身
            else:
                raise TypeError()

        super(KSSItemBase, self).__init__(
            items, knowledge_structure=knowledge_structure,
        )
        self.knowledge2item = dict()
        for item in self.items:
            self.knowledge2item[item.knowledge] = item
