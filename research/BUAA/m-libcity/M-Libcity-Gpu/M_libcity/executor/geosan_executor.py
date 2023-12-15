import numpy as np
import os
from tqdm import tqdm
import time as Time
from executor.abstract_executor import AbstractExecutor
import random
from utils import get_evaluator,getRelatedPath
import mindspore


class GeoSANExecutor(AbstractExecutor):

    def __init__(self, config, model, data_feature):
        self.config = config
        self.model = model
        self.evaluator = get_evaluator(config)
        self.exp_id = self.config.get('exp_id', None)
        self.cache_dir = getRelatedPath('cache/{}/model_cache'.format(self.exp_id))
        self.evaluate_res_dir = getRelatedPath('cache/{}/evaluate_cache'.format(self.exp_id))
        self.tmp_path = getRelatedPath('tmp/checkpoint/')

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(Dataloader): Dataloader
            eval_dataloader(Dataloader): None
        """

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(Dataloader): Dataloader
        """


    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        self.model.load(cache_name)

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.model.save(cache_name)

    @staticmethod
    def reset_random_seed(seed):
        """
        重置随机数种子

        Args:
            seed(int): 种子数
        """
        random.seed(seed)
        np.random.seed(seed)
        mindspore.set_seed(seed)
