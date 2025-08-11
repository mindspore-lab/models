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

from collections import defaultdict
from longling.ML.toolkit.monitor import EMAValue

STEP = 10
EPISODE = 20
SUMMARY = 30

str2level = {
    "step": STEP,
    "episode": EPISODE,
    "summary": SUMMARY,
}


def as_level(obj):
    if isinstance(obj, int):
        return_thing = obj
    else:
        return_thing = str2level[obj]
    return return_thing


def meta_train_eval(env_input_dict):
    """

    Parameters
    ----------
    agent
    env
    max_steps:
        When max_steps is set (i.e., max_steps is not None):
        at each episode, the agent will interactive at maximum of max_steps with environments.
        When max_steps is not set (i.e., max_steps is None): the episode will last until
        the environment return done=True

    max_episode_num:
        max_episode_num should be set when environment is the type of infinity
    n_step
    train
    logger
    values
    monitor

    Returns
    -------

    """
    agent = env_input_dict['agent']
    # 写一段代码让下面这几个元素全都等于env_input_dict里面的元素
    env = env_input_dict['env']
    max_steps = env_input_dict['max_steps']
    max_episode_num = env_input_dict['max_episode_num']
    n_step = env_input_dict['n_step']
    train = env_input_dict['train']
    logger = env_input_dict['logger']
    values = env_input_dict['values']
    level = env_input_dict['level']

    episode = 0

    level = as_level(level)

    rewards = []
    infos = []

    if values is None:
        values = {"Episode": EMAValue(["Reward"])}  # EMA is a technical chart indicator that tracks the price of an
        # investment overtime

    loop = max_episode_num
    logs_ = defaultdict(lambda: {})

    for i in range(loop):  # 一个episode
        if max_episode_num is not None and episode >= max_episode_num:
            # 如果设置了max_episode_num并且已经超过,就跳出循环
            break
        # begin episode
        try:
            learner_profile = env.begin_episode()  # 获得learner的初状态
            agent.begin_episode(learner_profile)
            episode += 1
            if level <= as_level("episode"):
                logger.info("episode [%s]: %s" % (episode, env.render("log")))

        except StopIteration:  # pragma: no cover
            break

        step_ = 0
        logs_[i * 3][0] = step_

        # recommend and learn
        if n_step is True:
            assert max_steps is not None
            # generate a learning path
            learning_path = agent.n_step(max_steps)  # agent一次直接取n个之后的learning item
            for observation, reward, done, info in env.n_step(learning_path):
                agent.observe(observation, reward, done, info)
                if done:
                    break
        else:
            # n_step = false就一步一步得产生learning item
            learning_path = []
            if max_steps is not None:  # max_steps=20
                # generate a learning path step by step
                for _ in range(max_steps):
                    try:
                        learning_item = agent.step()  # agent随机在action space中取下一个learning_item
                        learning_path.append(learning_item)
                    except StopIteration:  # pragma: no cover
                        break
                    observation, reward, done, info = env.step(learning_item)
                    agent.observe(observation, reward, done, info)
                    if done:
                        break
            else:
                raise ValueError("max_steps should be set when n_step is False")
        # 计算一个episode的reward
        observation, reward, done, info = env.end_episode()
        # agent把这一episode的学习情况记录下来，用于下一episode训练LSTM
        if not info:
            info = {}
        print(f"episode_reward: {reward}")
        agent.end_episode(observation, reward, done, info)
        # 消息记录
        rewards.append(reward)
        infos.append(info)
        # monitor更新显示信息，显示的是一个episode最终的reward
        values["Episode"].update("Reward", reward)
        # 清除当前learner
        env.reset()
        if train is True:
            agent.tune()


def train_eval(env_input_dict):
    max_episode_num = env_input_dict['max_episode_num']
    assert max_episode_num is not None, "infinity environment, max_episode_num should be set"

    meta_train_eval(env_input_dict=env_input_dict)
