import os
import time
import numpy as np

from mindspore import Callback

import numpy as np
from mindspore.common.tensor import Tensor
from model import loss
from mindspore import  ops
from mindspore import nn
from functools import partial
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore import Tensor, load_checkpoint, LossMonitor, TimeMonitor, CheckpointConfig, \
    ModelCheckpoint, Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.nn.metrics import MAE
from utils import get_evaluator, ensure_dir,getRelatedPath
from executor.traffic_state_executor import TrafficStateExecutor
from logging import getLogger
def nelement(param):
    ans=1
    for i in param:
        ans*=i
    return ans
class FOGSExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.use_trend = config.get('use_trend', True)
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.horizon = config.get('output_window', 1)
        self.trend_embedding = config.get('trend_embedding', False)
        self.output_window = config.get('output_window', 1)
        if self.trend_embedding:
            self.trend_bias_embeddings = nn.Embedding(288, self.num_nodes * self.output_window)
        self.evaluator = get_evaluator(config)
        self.config = config
        self.data_feature = data_feature
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
        self.lr_decay = self.config.get('lr_decay', True)
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
        load_checkpoint(model_path,net=self.model)
        self._logger.info("Loaded model at {}".format(epoch))

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
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
                    self.milestones[i]*=100
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

    def adjust_output(self, output, valx, valy_slot):
        if self.use_trend:
            # 获取输出张量的维度信息
            B, T, N = output.shape

            # 逆标准化并调整形状
            x_truth = self._scaler.inverse_transform(valx)
            x_truth = ops.Reshape()(x_truth,(B, T, -1))

            # 调整维度顺序
            x_truth = ops.Transpose()(x_truth, (1, 0, 2))  # (B, T, N) -> (T, B, N)
            output = ops.Transpose()(output, (1, 0, 2))  # (B, T, N) -> (T, B, N)

            if self.trend_embedding:
                # 计算趋势偏置，假设self.trend_bias_embeddings是类的另一个方法
                bias = self.trend_bias_embeddings(valy_slot[:, 0])
                bias = ops.Reshape()(bias, (-1, self.num_nodes, self.horizon))  # (B, N * T) -> (B, N, T)
                # 调整维度顺序
                bias = ops.Transpose()(bias, (2, 0, 1))  # (B, N, T) -> (T, B, N)

                # 确保偏置在正确的设备上
                #bias = bias.to(self.device)

                # 应用趋势偏置
                predict = (1 + output) * x_truth[-1] + bias
            else:
                # 应用趋势但不使用偏置
                predict = (1 + output) * x_truth[-1]

            # 恢复维度顺序
            predict = ops.Transpose()(predict, (1, 0, 2))  # (T, B, N) -> (B, T, N)
        else:
            # 如果不使用趋势，直接逆标准化
            predict = self._scaler.inverse_transform(output)
        return predict


    def evaluate(self, test_dataloader):

        self.model.eval()
        y_truths = []
        y_preds = []
        for batch in test_dataloader.create_dict_iterator():
            valx = batch['X']
            valy = batch['y'][:, :, :, 0]
            valy_slot = batch['y_slot']

            output = self.model.predict(batch['X'], batch['y'], batch['x_slot'], batch['y_slot'])
            predict = self.adjust_output(output, valx, valy_slot)
            y_truths.append(valy.asnumpy())
            y_preds.append(predict.asnumpy())

        y_preds = np.concatenate(y_preds, axis=0)
        y_truths = np.concatenate(y_truths, axis=0)

        # 保存结果
        outputs = {'prediction': y_preds, 'truth': y_truths}
        filename = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '_' + \
                   self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
        np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)

        # 以下是假设的评估逻辑，您需要根据您的 evaluator 实现来填充
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

        # 初始化记录最小验证损失的变量，以及用于记录训练和评估时间的列表
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        train_time = []
        eval_time = []

        # 获取训练数据加载器中的批次总数
        num_batches = train_dataloader.get_dataset_size()
        # 记录批次数
        self._logger.info("num_batches:{}".format(num_batches))

        self.model.train()

        #多卡
        #model = Model(self.model, optimizer=self.optimizer)  # self.model此时应返回loss
        #单卡
        # 创建模型实例，包括网络、优化器等配置
        model = Model(self.model, optimizer=self.optimizer) # self.model此时应返回loss

        # 创建损失监控的回调函数
        loss_cb = LossMonitor()
        # 创建时间监控的回调函数，记录每一步的时间
        time_cb = TimeMonitor(data_size=num_batches)
        # 创建检查点配置，设置保存检查点的间隔和最大保存数量
        config_ck = CheckpointConfig(save_checkpoint_steps=num_batches,
                                     keep_checkpoint_max=self.epochs,)
        # 创建模型检查点回调，用于保存训练过程中的模型
        ckpoint_cb = ModelCheckpoint(prefix=self.config['model']+'_'+self.config['dataset'], directory=self.cache_dir, config=config_ck)
        # 初始化评估相关参数
        per_eval = {"epoch": [], "loss": [],"best_epoch": 1,"min_val_loss": min_val_loss}
        #单卡
        eval_cb=FOGSEvalCallBack(self,self.model,eval_dataloader,1,per_eval,logger=self._logger,optim=self.optimizer)

        #多卡
        # ckpt_config = CheckpointConfig()
        # eval_cb = ModelCheckpoint(prefix='auto_parallel', config=ckpt_config)

        # 将所有回调函数加入到callbacks列表
        callbacks = [time_cb, loss_cb, ckpoint_cb,eval_cb]

        #单卡
        model.train(self.epochs,train_dataloader,callbacks=callbacks)

        #多卡
        #model.train(self.epochs,train_dataloader,callbacks=callbacks, dataset_sink_mode=True)

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


class FOGSEvalCallBack(Callback):
    """
        默认validation程序，此方法监控的metrics为loss
    """

    def __init__(self, executor, model, eval_dataloader, epochs_to_eval, per_eval, logger, optim=None):
        super(FOGSEvalCallBack, self).__init__()
        self.executor = executor  # 存储 FOGSExecutor 的实例
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.epochs_to_eval = epochs_to_eval
        self.per_eval = per_eval
        self._logger = logger
        self.optim = optim


    def on_train_epoch_end(self, run_context):
        if self.optim != None:
            print(self.optim.get_lr().asnumpy())
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        losses = []
        for batch in self.eval_dataloader.create_dict_iterator():
            valx = batch['X']
            valy =  batch['y'][:, :, :, 0]
            valy_slot =batch['y_slot']
            output = self.model.predict(batch['X'], batch['y'], batch['x_slot'], batch['y_slot'])
            predict = self.executor.adjust_output(output, valx, valy_slot)

            val_loss = loss.masked_mae_m(predict, valy, 0.0)
            losses.append(val_loss.asnumpy())

        # 计算平均损失
        mean_loss = np.mean(losses)
        self._logger.info("Epoch: {}, valid loss: {}".format(cur_epoch, mean_loss))

