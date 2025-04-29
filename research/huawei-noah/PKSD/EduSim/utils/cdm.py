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

__all__ = ["irt"]

import math
import numpy as np


def irt(ability, difficulty, discrimination=5, c=0.25):
    """
    Examples
    --------
    >>> round(irt(3, 1), 2)
    1.0
    >>> round(irt(1, 5), 3)
    0.25
    """
    return c + (1 - c) / (1 + math.exp(-1.7 * discrimination * (ability - difficulty)))


def dina(abilities, guessing, skipping):
    """
    Examples
    --------
    >>> dina([1, 1, 1], 0, 0)
    1
    >>> dina([0, 0], 1, 0)
    1
    >>> dina([1, 1], 1, 1)
    0
    >>> "%.2f" % dina([0.5, 0.5], 0.2, 0.2)
    '0.28'
    """
    eta = np.prod(abilities)
    return guessing ** (1 - eta) * (1 - skipping) ** eta
