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
from longling import json_load, path_append, abs_current_dir
from EduSim.Envs.shared.KSS_KES import KS
from EduSim.utils.io_lib import load_ks_from_csv
from EduSim.utils import get_proj_path

"""
Example
-------
>>> load_item("filepath_that_do_not_exsit/not_exsit77250")
{}
"""


def load_items(filepath):
    if os.path.exists(filepath):
        return json_load(filepath)
    else:
        return {}


def load_knowledge_structure(filepath):
    knowledge_structure = KS()
    knowledge_structure.add_edges_from([list(map(int, edges)) for edges in load_ks_from_csv(filepath)])
    return knowledge_structure


def load_learning_order(filepath):
    return json_load(filepath)


def load_configuration(filepath):
    return json_load(filepath)


def load_environment_parameters(directory=None):
    if directory is None:
        directory = f'{get_proj_path()}/EduSim/Envs/KSS/meta_data'
    return {
        "configuration": load_configuration(path_append(directory, "configuration.json")),
        "knowledge_structure": load_knowledge_structure(path_append(directory, "knowledge_structure.csv")),
        "learning_order": load_learning_order(path_append(directory, "learning_order.json")),
        "items": load_items(path_append(directory, "items.json"))
    }
