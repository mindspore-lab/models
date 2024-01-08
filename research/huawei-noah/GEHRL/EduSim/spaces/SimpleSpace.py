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
from pprint import pformat
from gym.spaces import Space

__all__ = ["ListSpace"]


class ListSpace(Space):
    def __init__(self, elements: list, seed=None):
        self.elements = elements
        super(ListSpace, self).__init__(shape=(len(self.elements),))
        self.seed(seed)

    def sample(self):
        return self.np_random.choice(self.elements)

    def sample_idx(self):
        return self.np_random.choice(list(range(len(self.elements))))

    def contains(self, item):
        return item in self.elements

    def __repr__(self):
        return pformat(self.elements)

    def __getitem__(self, item):
        return self.elements[item]
