import os
import time
import numpy as np
import mindspore.nn as nn
from logging import getLogger
from mindspore import Tensor, load_checkpoint, LossMonitor, TimeMonitor, CheckpointConfig, \
    ModelCheckpoint, Model
from functools import partial
from executor.abstract_executor import AbstractExecutor
from executor.utils import EvalCallBack
from model import loss
import mindspore as ms

from utils import get_evaluator, ensure_dir,getRelatedPath
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from executor.softmax_cross_entropy_expand import SoftmaxCrossEntropyExpand

def nelement(param):
    ans=1
    for i in param:
        ans*=i
    return ans


class TrafficStateExecutor(AbstractExecutor):
    def __init__(self, config, model,data_feature):
        self.evaluator = get_evaluator(config)
        self.config = config
        self.data_feature = data_feature
        # self.model = model.to(self.device)
        self.model=model

        self.exp_id = self.config.get('exp_id', None)

        self.cache_dir = getRelatedPath('cache/{}/model_cache'.format(self.exp_id))
        self.evaluate_res_dir = getRelatedPath('cache/{}/evaluate_cache'.format(self.exp_id))
        self.summary_writer_dir = getRelatedPath('cache/{}/'.format(self.exp_id))
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)

        # self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self._logger.info(self.model)
        for param in self.model.trainable_params():
            self._logger.info(str(param.name) + '\t' + str(param.shape) + '\t' +
                              str(param.requires_grad))
        total_num = sum([nelement(param.shape) for param in self.model.trainable_params()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        self.epochs = self.config.get('max_epoch', 100)
        self.train_loss = self.config.get('train_loss', 'none')
        self.learner = self.config.get('learner', 'adam')
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.weight_decay = self.config.get('weight_decay', 0)
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = self.config.get('lr_alpha', 0.99)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.lr_momentum = self.config.get('lr_momentum', 0)
        self.lr_decay = self.config.get('lr_decay', False)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)
        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.saved = self.config.get('saved_model', True)
        self.load_best_epoch = self.config.get('load_best_epoch', True)
        # self.hyper_tune = self.config.get('hyper_tune', False)

        self.output_dim = self.config.get('output_dim', 1)

        # 在mindspore中应先计算scheduler，再创建optimizer
        self.lr_scheduler = self._build_lr_scheduler()
        self.optimizer = self._build_optimizer()

        self._epoch_num = self.config.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model_with_epoch(self._epoch_num)
        self.loss_func = self._build_train_loss()



    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at " + cache_name)
        load_checkpoint(cache_name,net=self.model)
        # load_param_into_net(self.model,model_state)

    def load_model_with_epoch(self, epoch, num_batch):
        """
        加载某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '-{}_{}.ckpt'.format(epoch, num_batch)
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        load_checkpoint(model_path,net=self.model)  # fixme 加载模型
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        # print('!!!', self.model.trainable_params(), self.lr_scheduler)
        
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.lr_scheduler,
                                         eps=self.lr_epsilon, beta1=self.lr_beta1,beta2=self.lr_beta2, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = nn.SGD(self.model.trainable_params(), learning_rate=self.lr_scheduler,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = nn.Adagrad(self.model.trainable_params(), learning_rate=self.lr_scheduler,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = nn.RMSProp(self.model.trainable_params(), learning_rate=self.lr_scheduler,
                                            decay=self.lr_alpha, epsilon=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
      
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.lr_scheduler,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':

                self.lr_list=[self.learning_rate]
                for i in range(len(self.milestones)):
                    self.milestones[i]*=1000
                for _ in range(len(self.milestones)-1):
                    self.lr_list.append(self.lr_list[-1]*self.lr_decay_ratio)
                lr_scheduler = nn.piecewise_constant_lr(milestone=self.milestones, learning_rates=self.lr_list)

            elif self.lr_scheduler_type.lower() == 'steplr':
                self.milestones=[self.step_size*1000]
                self.lr_list=[self.learning_rate]

                while self.milestones[-1] <(self.epochs*1000):
                    self.milestones.append(self.milestones[-1]*2)
                    self.lr_list.append(self.lr_list[-1]*self.lr_decay_ratio)
                lr_scheduler = nn.piecewise_constant_lr(milestone=self.milestones, learning_rates=self.lr_list)

            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = nn.exponential_decay_lr(learning_rate=self.learning_rate,decay_rate=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = nn.dynamic_lr.cosine_decay_lr( max_lr=self.lr_T_max, min_lr=self.lr_eta_min)


            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = self.learning_rate
        else:
            lr_scheduler = self.learning_rate
        return lr_scheduler

    def _build_train_loss(self):
        """
        根据全局参数`train_loss`选择训练过程的loss函数
        如果该参数为none，则需要使用模型自定义的loss函数
        注意，loss函数应该接收`Batch`对象作为输入，返回对应的loss(tensor)
        """

        if self.train_loss.lower() == 'none':
            self._logger.warning('Received none train loss func and will use the loss func defined in the model.')
            return None
        if self.train_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile',
                                           'masked_mae',
                                           'masked_mse', 'masked_rmse', 'masked_mape', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        else:
            self._logger.info('You select `{}` as train loss function.'.format(self.train_loss.lower()))


        if self.train_loss.lower() == 'mae':
            lf = loss.masked_mae_m
        elif self.train_loss.lower() == 'mse':
            lf = loss.masked_mse_m
        elif self.train_loss.lower() == 'rmse':
            lf = loss.masked_rmse_m
        elif self.train_loss.lower() == 'mape':
            lf = loss.masked_mape_m
        elif self.train_loss.lower() == 'logcosh':
            lf = loss.log_cosh_loss
        elif self.train_loss.lower() == 'huber':
            lf = loss.huber_loss
        elif self.train_loss.lower() == 'quantile':
            lf = loss.quantile_loss
        elif self.train_loss.lower() == 'masked_mae':
            lf = partial(loss.masked_mae_m, null_val=0)
        elif self.train_loss.lower() == 'masked_mse':
            lf = partial(loss.masked_mse_m, null_val=0)
        elif self.train_loss.lower() == 'masked_rmse':
            lf = partial(loss.masked_rmse_m, null_val=0)
        elif self.train_loss.lower() == 'masked_mape':
            lf = partial(loss.masked_mape_m, null_val=0)
        elif self.train_loss.lower() == 'r2':
            lf = loss.r2_score_m
        elif self.train_loss.lower() == 'evar':
            lf = loss.explained_variance_score_m
        else:
            lf = loss.masked_mae_m

        return lf


    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')

        # test_dataloader=test_dataloader.split([0.1,0.9])[0]
        y_truths = []
        y_preds = []
        self.model.eval()  # 该步对应操作为self.model.mode="eval",让model返回y_pred和y_true
        
        for batch in test_dataloader:
            y_pred,y_true = self.model.evaluate(*batch)
            y_true = y_true[..., : self.output_dim]  # fixme 确认一下是否正确
            y_pred = y_pred[..., : self.output_dim]
            y_truths.append(y_true.asnumpy().astype(np.float32))
            y_preds.append(y_pred.asnumpy().astype(np.float32))

        self.model.train()  # 将model状态调回训练模式

        y_preds = np.concatenate(y_preds, axis=0)
        y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
        print(y_preds.shape)
        print(y_truths.shape)
        outputs = {'prediction': y_preds, 'truth': y_truths}
        filename = \
            time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
            + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
        np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
        self.evaluator.clear()
        self.evaluator.collect({'y_true': Tensor(y_truths), 'y_pred': Tensor(y_preds)})
        test_result = self.evaluator.save_result(self.evaluate_res_dir)
        return test_result

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(Dataloader): Dataloader
            eval_dataloader(Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        train_time = []
        eval_time = []
        # train_dataloader=train_dataloader.split([0.01,0.99])[0]
        # eval_dataloader = eval_dataloader.split([0.1, 0.9])[0]

        num_batches = train_dataloader.get_dataset_size()


        self._logger.info("num_batches:{}".format(num_batches))

        # self.model.set_loss(self.loss_func)  # 定义loss

        self.model.train() #该步对应操作为self.model.mode="train"

        #多卡
        # loss_fn = SoftmaxCrossEntropyExpand(sparse=True,logger=self._logger)
        # model = ms.amp.build_train_network(self.model, optimizer=self.optimizer)  # self.model此时应返回loss
        # loss_scale_manager = ms.DynamicLossScaleManager()
        # model = Model(self.model, loss_scale_manager=loss_scale_manager,optimizer=self.optimizer,amp_level="O3") 
        #单卡
        model = Model(self.model, optimizer=self.optimizer)  # self.model此时应返回loss

        loss_cb = LossMonitor()
        time_cb = TimeMonitor(data_size=num_batches)
        config_ck = CheckpointConfig(save_checkpoint_steps=num_batches,
                                     keep_checkpoint_max=self.epochs,)
        ckpoint_cb = ModelCheckpoint(prefix=self.config['model']+'_'+self.config['dataset'], directory=self.cache_dir, config=config_ck)
        per_eval = {"epoch": [], "loss": [],"best_epoch": 0,"min_val_loss": min_val_loss}
        #单卡
        eval_cb=EvalCallBack(self.model,eval_dataloader,1,per_eval,logger=self._logger,optim=self.optimizer)
        #多卡
        # ckpt_config = CheckpointConfig()
        # eval_cb = ModelCheckpoint(prefix='auto_parallel', config=ckpt_config)
        callbacks = [time_cb, loss_cb, ckpoint_cb,eval_cb]
        #单卡
        #model.train(self.epochs,train_dataloader,callbacks=callbacks)
        #多卡
        model.train(self.epochs,train_dataloader,callbacks=callbacks, dataset_sink_mode=True)
        # model.train(1,train_dataloader,callbacks=callbacks, dataset_sink_mode=True)
        # model.train(1,train_dataloader,callbacks=callbacks)

        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            best_epoch=per_eval['best_epoch']
            best_model_name = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '-{}_{}.ckpt'.format(best_epoch, num_batches)
            target_model_name = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset']+'.ckpt'
            os.rename(best_model_name,target_model_name)

            self.load_model(target_model_name)

        return per_eval['min_val_loss']



