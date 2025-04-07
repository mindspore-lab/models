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
import random
from EduSim.Envs.meta import TraitScorer


class KESScorer(TraitScorer):
    def response_function(self, user_trait, item_trait, *args, **kwargs):
        # return 1 if user_trait[item_trait] >= 0.55 else 0
        # return 1 if user_trait[item_trait] >= 0.67 else 0
        # return 1 if user_trait[item_trait] >= 0.73 else 0
        return 1 if user_trait[item_trait] >= 0.75 else 0  # original

    def middle_response_function(self, user_trait, item_trait, *args, **kwargs):
        return 1 if random.random() <= user_trait[item_trait] else 0
