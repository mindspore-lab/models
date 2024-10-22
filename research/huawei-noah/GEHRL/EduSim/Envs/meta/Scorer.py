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


class Scorer:
    def __call__(self, user, item, *args, **kwargs) -> ...:
        raise NotImplementedError


class RealScorer(Scorer):
    def answer_scoring(self, user_response, item_truth, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, user_response, item_truth, *args, **kwargs):
        return self.answer_scoring(user_response, item_truth, *args, **kwargs)


class RealChoiceScorer(RealScorer):
    def answer_scoring(self, user_response, item_truth, *args, **kwargs):
        return user_response == item_truth


class TraitScorer(Scorer):
    def response_function(self, user_trait, item_trait, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, user_trait, item_trait, *args, **kwargs):
        return self.response_function(user_trait, item_trait, *args, **kwargs)
