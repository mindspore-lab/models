# Copyright 2024 Xidian University
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
import numpy as np


def distance_score(delta_1, delta_2, mode = 'IOU', L=640.):
    if mode == 'distance':
        distance_1 = np.sqrt(delta_1[:, 0] * delta_1[:, 0] + delta_1[:, 1] * delta_1[:, 1])
        distance_2 = np.sqrt(delta_2[:, 0] * delta_2[:, 0] + delta_2[:, 1] * delta_2[:, 1])
        ratio = distance_1/distance_2
    elif mode == 'IOU':
        IOU_1 = 1. / (1. - (1 - np.abs(delta_1[:, 0]) / L) * (1. - np.abs(delta_1[:, 1]) / L) / 2.) - 1
        IOU_2 = 1. / (1. - (1 - np.abs(delta_2[:, 0]) / L) * (1. - np.abs(delta_2[:, 1]) / L) / 2.) - 1
        ratio = IOU_2/ IOU_1
    return ratio
