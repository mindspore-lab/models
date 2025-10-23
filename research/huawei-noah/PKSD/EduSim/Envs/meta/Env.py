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
import gym



class Env(gym.Env):
    metadata = {'render.modes': ['human', 'log']}

    def __repr__(self):
        return pformat(self.parameters)

    @property
    def parameters(self) -> dict:
        return {}

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        return ""

    def step(self, learning_item_id, *args, **kwargs):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            learning_item_id (object): an learning item id provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """
        raise NotImplementedError

    def n_step(self, learning_path, *args, **kwargs):
        raise NotImplementedError

    def begin_episode(self, *args, **kwargs):
        """

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        learner_profile

        """
        raise NotImplementedError

    def end_episode(self, *args, **kwargs):
        raise NotImplementedError
