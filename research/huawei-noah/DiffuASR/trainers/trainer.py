# -*- encoding: utf-8 -*-
# here put the import lib
import os
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context
from tqdm import tqdm, trange
from utils.earlystop import EarlyStoppingNew
from models.Bert4Rec import Bert4Rec


class Trainer(object):
    def __init__(self, args, logger, generator):
        self.args = args
        self.logger = logger
        self.user_num, self.item_num = generator.get_user_item_num()
        self.start_epoch = 0

        self.logger.info('Loading Model: ' + args.model_name)
        self._create_model()
        logger.info('# of model parameters: ' + str(self.get_n_params(self.model)))

        self._set_optimizer()
        self._set_stopper()

        if args.keepon:
            self._load_pretrained_model()

        self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')

        self.train_loader = generator.make_trainloader()
        self.valid_loader = generator.make_evalloader()
        self.test_loader = generator.make_evalloader(test=True)
        self.generator = generator

        self.watch_metric = args.watch_metric

    def _create_model(self):
        if self.args.model_name == 'bert4rec':
            self.model = Bert4Rec(self.user_num, self.item_num, self.args)
        else:
            raise ValueError

    def _load_pretrained_model(self):
        self.logger.info("Loading the trained model for keep on training ... ")
        checkpoint_path = os.path.join(self.args.keepon_path, 'mindspore_model.ckpt')
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(self.model, param_dict)

    def _set_optimizer(self):
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.args.lr, weight_decay=self.args.l2)

    def _set_stopper(self):
        self.stopper = EarlyStoppingNew(patience=self.args.patience, verbose=False, path=self.args.output_dir)

    def _train_one_epoch(self, epoch):
        # Your training logic goes here
        return NotImplementedError

    def eval(self, epoch=0, test=False):
        # Your evaluation logic goes here
        return NotImplementedError

    def train(self):
        model_to_save = self.model
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        res_list = []
        train_time = []

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.num_train_epochs), desc="Epoch"):
            t = self._train_one_epoch(epoch)
            train_time.append(t)

            if (epoch % 1) == 0:
                metric_dict = self.eval(epoch=epoch)
                res_list.append(metric_dict)
                self.stopper(metric_dict[self.watch_metric], epoch, model_to_save, self.optimizer)

                if self.stopper.early_stop:
                    break

        best_epoch = self.stopper.best_epoch
        best_res = res_list[best_epoch - self.start_epoch]
        self.logger.info('')
        self.logger.info('The best epoch is %d' % best_epoch)
        self.logger.info('The best results are NDCG@10: %.5f, HR@10: %.5f' %
                         (best_res['NDCG@10'], best_res['HR@10']))

        res = self.eval(test=True)

        return res, best_epoch

    def get_model(self):
        return self.model

    def get_n_params(self, model):
        total_params = 0
        for param in model.get_parameters():
            total_params += param.size
        return total_params

