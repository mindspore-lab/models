import os
from logging import getLogger

from mindspore import load_checkpoint, load_param_into_net, Model, LossMonitor, TimeMonitor, CheckpointConfig, \
    ModelCheckpoint
import mindspore.nn as nn
import mindspore
from tqdm import tqdm
import time

from executor.abstract_executor import AbstractExecutor
from executor.utils import EvalCallBack
from utils import get_evaluator,getRelatedPath


class TrajLocPredExecutor(AbstractExecutor):

    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config)
        self.metrics = 'Recall@{}'.format(config['topk'])
        self.config = config
        self.model=model
        self.exp_id = self.config.get('exp_id', None)
        self.cache_dir = getRelatedPath('cache/{}/model_cache'.format(self.exp_id))
        self.evaluate_res_dir = getRelatedPath('cache/{}/evaluate_cache'.format(self.exp_id))
        self.loss_func = mindspore.ops.NLLLoss() 
        self._logger = getLogger()
        self.learning_rate=self.config['learning_rate']
        self.learner=self.config['optimizer']
        self.load_best_epoch = self.config.get('load_best_epoch', True)

        self.epochs=self.config['max_epoch']

        self.lr_scheduler=0.001
        self.optimizer = self._build_optimizer()

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(Dataloader): Dataloader
            eval_dataloader(Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        best_acc = 0
        train_time = []
        eval_time = []

        num_batches = train_dataloader.get_dataset_size()

        self._logger.info("num_batches:{}".format(num_batches))

        self.model.set_loss(self.loss_func)  # 定义loss

        self.model.train()  # 该步对应操作为self.model.mode="train"
        # ms的混合精度，代码备份
        # mindspore.amp.auto_mixed_precision(self.model, amp_level='O2')
        # model = Model(self.model, optimizer=self.optimizer)  # self.model此时应返回loss
        # mindspore.amp.auto_mixed_precision(model, amp_level='O3')

        # loss_cb = LossMonitor()
        # time_cb = TimeMonitor(data_size=num_batches)
        # config_ck = CheckpointConfig(save_checkpoint_steps=num_batches,
        #                              keep_checkpoint_max=self.epochs, )
        # ckpoint_cb = ModelCheckpoint(prefix=self.config['model'] + '_' + self.config['dataset'],
        #                              directory=self.cache_dir, config=config_ck)
        per_eval = {"epoch": [], "best_epoch": 0, "best_acc": best_acc}

        def foward_fn(batch):
            loss = self.model(*batch)
            return loss,None

        grad_fn=mindspore.value_and_grad(foward_fn,None,weights=self.optimizer.parameters,has_aux=True)

        for epoch in range(self.epochs):
            
            total = train_dataloader.get_dataset_size()
            loss_total = 0
            step_total = 0
            start_time=int(time.time())
            with tqdm(total=total) as t:
                t.set_description('Epoch %i' % epoch)
                for indx,batch in enumerate(train_dataloader):
                    (loss, _), grads = grad_fn(batch)
                    grads = mindspore.ops.clip_by_value(grads, clip_value_max=1.0)
                    loss = mindspore.ops.depend(loss, self.optimizer(grads))
                    
                    if (indx+1) % 10 == 0:
                        loss, current = loss, indx
                        loss_total += loss
                        step_total += 1

                        t.set_postfix(loss_t=loss_total/step_total)
                        t.update(10)
                        
            total_loss = loss_total/step_total
            end_time=int(time.time())
            tt=end_time-start_time
            train_time.append(tt)
            start_time=end_time
            valid_acc = self.validation(eval_dataloader)
            end_time=int(time.time())
            tt=end_time-start_time
            eval_time.append(tt)

            print("-" * 50)
            print("Epoch: [%3d/%3d], Average Train Loss: [%5.3f], Accuracy: [%5.3f]" % (
                epoch+1, self.epochs, total_loss, valid_acc
            ))
            self._logger.info("Epoch: [%3d/%3d], Average Train Loss: [%5.3f], Accuracy: [%5.3f]" % (
                epoch+1, self.epochs, total_loss, valid_acc
            ))
            print("-" * 50)
            if valid_acc > per_eval['best_acc']:
                per_eval['best_acc'] = valid_acc
                target_model_name = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '.ckpt'
                if not os.path.exists(self.cache_dir):
                    os.makedirs(self.cache_dir)
                print("save model in"+target_model_name)
                self._logger.info("save model in"+target_model_name)
                model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
                mindspore.save_checkpoint(self.model, target_model_name)


        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model(target_model_name)

        return per_eval['best_acc']


    def load_model(self, cache_name):
        self._logger.info("Loaded model at " + cache_name)
        load_checkpoint(cache_name,net=self.model)

    def validation(self, test_dataset):
        correct_num = 0.0  # 预测正确个数
        total_num = 0.0  # 预测总数
        self.model.eval()

        for batch in test_dataset.create_dict_iterator():
            scores = self.model(*batch.values())
            pred = scores.argmax(axis=1)
            correct = mindspore.ops.equal(pred, batch['target']).reshape((-1, ))
            correct_num += correct.sum().asnumpy()
            total_num += correct.shape[0]

        self.model.train()
        acc = correct_num / total_num

        return acc

    def evaluate(self, test_dataloader):
        self.model.eval()
        self.evaluator.clear()
        for batch in test_dataloader.create_dict_iterator():
            scores = self.model(*batch.values())
            if self.config['evaluate_method'] == 'popularity':
                evaluate_input = {
                    'uid': batch['uid'],
                    'loc_true': batch['target'],
                    'loc_pred': scores
                }
            else:
                loc_true = [0] * self.config['batch_size']
                evaluate_input = {
                    'uid': batch['uid'],
                    'loc_true': loc_true,
                    'loc_pred': scores
                }
            self.evaluator.collect(evaluate_input)
        self.model.train()
        self.evaluator.save_result(self.evaluate_res_dir)

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.lr_scheduler,weight_decay=self.config['L2'])
        elif self.learner.lower() == 'sgd':
            optimizer = nn.SGD(self.model.trainable_params(), learning_rate=self.lr_scheduler,weight_decay=self.config['L2'])
        elif self.learner.lower() == 'adagrad':
            optimizer = nn.Adagrad(self.model.trainable_params(), learning_rate=self.lr_scheduler,weight_decay=self.config['L2'])
        elif self.learner.lower() == 'rmsprop':
            optimizer = nn.RMSProp(self.model.trainable_params(), learning_rate=self.lr_scheduler,weight_decay=self.config['L2'] )

        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.lr_scheduler,weight_decay=self.config['L2'])
        return optimizer


