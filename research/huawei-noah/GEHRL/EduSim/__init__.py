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
from gym.envs.registration import register
from .Envs import KESASSISTEnv
from .SimOS import train_eval
from .spaces import ListSpace
from .AbstractAgent import AbstractAgent
from .buffer import ReplayBuffer
from .deep_model import RnnEncoder, PolicyNet, ValueNet, PolicyNetWithOutterEncoder, ValueNetWithOutterEncoder, DKTnet

register(
    id='KES-v1',
    entry_point='EduSim.Envs:KESEnv'
)
register(
    id='KESASSIST-v1',
    entry_point='EduSim.Envs:KESASSISTEnv'
)
# register(
#     id='KSS-v2',
#     entry_point='EduSim.Envs:KSSEnv',
# )
# register(
#     id='KESASSIST09-v1',
#     entry_point='EduSim.Envs:KESASSIST09Env'
# )
