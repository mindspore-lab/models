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


class Splitter:
    def __init__(self):
        self.order = []
        self.part = dict()

    def add(self, name, weight):
        assert name not in self.order
        assert weight >= 0

        self.order.append(name)
        self.part[name] = weight
        return self

    def divide(self, amount):
        sum_weight = sum(self.part.values())
        assert sum_weight > 0

        range_dict = dict()

        start = 0
        for name in self.order[:-1]:
            end = int(start + self.part[name] / sum_weight * amount) + 1
            range_dict[name] = (start, end)
            start = end

        end = amount
        range_dict[self.order[-1]] = (start, end)
        return range_dict

    def contains(self, name):
        return name in self.part
