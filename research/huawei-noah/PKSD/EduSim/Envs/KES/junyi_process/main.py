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


__all__ = ["extract_relations", "build_json_sequence"]

import sys
import os
from longling import path_append
from EduSim.utils import get_proj_path, get_raw_data_path
from .junyi import build_knowledge_graph
from .KnowledgeTracing import select_n_most_frequent_students

sys.path.append(os.path.dirname(sys.path[0]))  # 这里要改为你自己的项目的主目录


def extract_relations(src_root: str = "../raw_data/junyi/", tar_root: str = "../data/junyi/"):
    build_knowledge_graph(
        src_root, tar_root,
        ku_dict_path="graph_vertex.json",
        prerequisite_path="prerequisite.json",
        similarity_path="similarity.json",
        difficulty_path="difficulty.json",
    )


def build_json_sequence(src_root: str = "../raw_data/junyi/", tar_root: str = "../data/junyi/",
                        ku_dict_path: str = "../data/junyi/graph_vertex.json", n: int = 1000):
    select_n_most_frequent_students(
        path_append(src_root, "junyi_ProblemLog_for_PSLC.txt", to_str=True),
        path_append(tar_root, "student_log_kt_", to_str=True),
        ku_dict_path,
        n,
    )


if __name__ == "__main__":
    src_root = f'{get_raw_data_path()}/junyi/'
    tar_root = f'{get_proj_path()}/data/dataProcess/junyi/'
    # extract_relations(src_root=src_root, tar_root=tar_root)
    build_json_sequence(src_root=src_root, tar_root=tar_root, ku_dict_path=tar_root + 'graph_vertex.json', n=None)
