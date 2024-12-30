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

from EduSim.utils import irt
from EduSim.Envs.meta import TraitScorer


class KSSScorer(TraitScorer):
    def __init__(self, binary_scorer=True):
        super(KSSScorer, self).__init__()
        self._binary = binary_scorer

    def response_function(self, user_trait, item_trait, binary=None, *args, **kwargs):
        _score = irt(user_trait, item_trait["difficulty"])  # IRT learner对knowledge的掌握情况和test_item的难度关系来打分
        binary = self._binary if binary is None else binary
        if binary:
            return 1 if _score >= 0.5 else 0
        else:
            return _score
