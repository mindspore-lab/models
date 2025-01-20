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

import os
import re

import yaml
from oba import Obj
from pigmento import pnt


class ConfigInitializer:
    @staticmethod
    def get_config_value(config: Obj, path: str):
        path = path.split('.')
        path_ = []
        for key in path:
            list_keys = key.split('[')
            for list_key in list_keys:
                if list_key.endswith(']'):
                    list_key = int(list_key[:-1])
                path_.append(list_key)

        value = config
        for key in path_:
            if type(key) != int:
                value = value[key]
        return value

    @classmethod
    def format_config_path(cls, config: Obj, path: str):
        dynamic_values = re.findall('{.*?}', path)
        for dynamic_value in dynamic_values:
            path = path.replace(dynamic_value, str(cls.get_config_value(config, dynamic_value[1:-1])))
        return path

    @classmethod
    def init(cls, config_path, exp_path):
        config = yaml.safe_load(open(config_path))
        config = Obj(config)

        exp = yaml.safe_load(open(exp_path))
        exp = Obj(exp)

        exp.model = exp.model.upper()
        pnt('model:', exp.model)

        meta_config = Obj(dict(exp=exp, config=config))

        if config.store.data_dir:
            config.store.data_dir = cls.format_config_path(meta_config, config.store.data_dir)
        config.store.save_dir = cls.format_config_path(meta_config, config.store.save_dir)
        exp.exp = cls.format_config_path(meta_config, exp.exp)
        if exp.load.load_ckpt:
            exp.load.load_ckpt = cls.format_config_path(meta_config, exp.load.load_ckpt)
        if exp.load.ckpt_base_path:
            exp.load.ckpt_base_path = cls.format_config_path(meta_config, exp.load.ckpt_base_path)

        # config.store.ckpt_path = os.path.join(config.store.save_dir, exp.exp)
        config.store.ckpt_path = 'saving/aotm-n10/BERT-E64/curriculum-bert-double-step'
        # config.store.log_path = os.path.join('saving/aotm-n10/BERT-E64/curriculum-bert-double-step', '{}.log'.format(exp.exp))
        #C:\Users\hwx1216117\PycharmProjects\FANS\FastCont\FastCont\saving\aotm-n10\BERT-E64\test-curriculum-bert-double-step\test-curriculum-bert-double-step.log
        config.store.log_path = 'saving/aotm-n10/BERT-E64/test-curriculum-bert-double-step/test-curriculum-bert-double-step.log'

        os.makedirs(config.store.ckpt_path, exist_ok=True)
        # os.makedirs('saving/aotm-n10/BERT-E64/curriculum-bert-double-step', exist_ok=True)

        return config, exp
