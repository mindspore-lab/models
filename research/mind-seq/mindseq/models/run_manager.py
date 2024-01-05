import os
import mindspore as ms
import logging
# pylint: disable=E0401
import mindseq.utils.metric as Metric
from .layers.mode import Mode
import time

# data_path = '../dataset'
# param_path = '../param'
class RunManager:
    def __init__(self,
                 path,
                 name,
                 net,
                 dataset,
                 arch_lr, arch_lr_decay_milestones, arch_lr_decay_ratio, arch_decay, arch_clip_gradient,
                 weight_lr, weight_lr_decay_milestones, weight_lr_decay_ratio, weight_decay, weight_clip_gradient,
                 num_search_epochs, num_train_epochs,
                 criterion, metric_names, metric_indexes,
                 print_frequency,
                 use_gpu, device_ids, reduce_flag):
        self.param_path = path
        self._name = name
        self._dataset = dataset
        self._net = self.net_mount_device(net, use_gpu, device_ids)

        # arch optimizer
        self._arch_lr = arch_lr
        self._arch_lr_decay_milestones = arch_lr_decay_milestones
        self._arch_lr_decay_ratio = arch_lr_decay_ratio
        self._arch_decay = arch_decay
        self._arch_clip_gradient = arch_clip_gradient

        # nn optimizer
        self._weight_lr = weight_lr
        self._weight_lr_decay_milestones = weight_lr_decay_milestones
        self._weight_lr_decay_ratio = weight_lr_decay_ratio
        self._weight_decay = weight_decay
        self._weight_clip_gradient = weight_clip_gradient

        self._num_search_epochs = num_search_epochs
        self._num_train_epochs = num_train_epochs

        self._criterion = getattr(Metric, criterion)
        self._metric_names = metric_names
        self._metric_indexes = metric_indexes
        self._print_frequency = print_frequency
        
        self.reduce_flag = reduce_flag
        
    def net_mount_device(self, net, use_gpu, device_ids):
        if use_gpu:
            use_devices = ','.join([str(d_id) for d_id in device_ids])
            os.environ["CUDA_VISIBLE_DEVICES"] = use_devices
        else:
            pass
        return net

    def _load(self, exp_mode):
        # initialize for optimizers and clear validation records
        self.initialize()
        save_dir = os.path.join(self.param_path, self._name)
        filename = os.path.join(save_dir, '{}.ckpt'.format(exp_mode))
        try:
            states = ms.load_checkpoint(filename)
            # load net
            self._net.load_state_dict(states['net'])
            # load optimizer
            self._arch_optimizer.load_state_dict(states['arch_optimizer'])
            self._weight_optimizer.load_state_dict(states['weight_optimizer'])
            # load historical records
            self._best_epoch = states['best_epoch']
            self._valid_records = states['valid_records']
            # logging.info('load architecture [epoch {}] from {} [ok]'.format(self._best_epoch, filename))
            print('load architecture [epoch {}] from {} [ok]'.format(self._best_epoch, filename))
        except:
            # logging.info('load architecture [fail]')
            # logging.info('initialize the optimizer')
            print('load architecture [fail]')
            print('initialize the optimizer')

            self.initialize()

    def clear_records(self):
        self._best_epoch = -1
        self._valid_records = []

    def initialize(self):
        # initialize for weight optimizer
        self._weight_optimizer = ms.nn.Adam(
            self._net.weight_parameters(),
            learning_rate=self._weight_lr,
            weight_decay=self._weight_decay
        )
        # initialize for arch optimizer
        self._arch_optimizer = ms.nn.Adam(
            self._net.arch_parameters(),
            learning_rate=self._arch_lr,
            weight_decay=self._arch_decay
        )
        # initialize validation records 
        self.clear_records()

    
    def _train_epoch(self, epoch, train_loader, tag='train', net_mode=Mode.ALL_PATHS, weights=None):
        speedometer = Speedometer(
            title=tag,
            epoch=epoch,
            metric_names=self._metric_names,
            metric_indexes=self._metric_indexes,
            print_frequency=self._print_frequency,
            batch_size=self._dataset._batch_size
        )
        def forward_fn(batch_x, batch_x_mark, adj_mats, mode, weights):
            preds = self._net(batch_x, batch_x_mark, attn_mask = None, 
                            adj_mats = adj_mats, mode = mode, weights = weights)
            preds = self._dataset.scaler.inverse_transform(preds)
            loss = self._criterion(preds, batch_y)
            return loss, preds
        
        def clip_grad_norm(grads, max_norm=1.0, norm_type=2):
            total_norm = 0
            for p in grads:
                param_norm = ms.ops.flatten(p).norm(norm_type).asnumpy()
                total_norm += param_norm.item() ** norm_type
            total_norm = total_norm ** (1. / norm_type)
            clip_coef = max_norm / (total_norm + 1e-6)
            return min(clip_coef, 1)
        
        if self.reduce_flag:
            mean = ms.context.get_auto_parallel_context("gradients_mean")
            degree = ms.context.get_auto_parallel_context("device_num")
            grad_reducer = ms.nn.DistributedGradReducer(self._weight_optimizer.parameters, mean, degree)
        
        cast = ms.ops.Cast()
        self._net.set_train()
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader.create_tuple_iterator()):
            batch_x = cast(batch_x, ms.float32)
            batch_x_mark = cast(batch_x_mark, ms.float32)
            batch_y = cast(batch_y, ms.float32)
            grad_fn = ms.ops.value_and_grad(forward_fn, None, self._weight_optimizer.parameters, has_aux=True)
            (loss, preds), grads = grad_fn(batch_x, batch_x_mark, ms.Tensor.from_numpy(self._dataset.adj_mats).float(), net_mode, weights)
            if self.reduce_flag:
                grads = grad_reducer(grads)
            loss = ms.ops.depend(loss, self._weight_optimizer(grads))
            speedometer.update(preds, batch_y)
        # logging.info('-'*30)
        # logging.info('Train epoch [{}] finished'.format(epoch))
        # logging.info('-'*30)
        print('-'*30)
        print('Train epoch [{}] finished'.format(epoch))
        print('-'*30)

        return speedometer.finish()

    def train(self, num_train_epochs=None, net_mode=Mode.ONE_PATH_FIXED, rank_id=0):
        self.clear_records()

        train_loader = self._dataset.get_dataloader(tag='train')
        valid_loader = self._dataset.get_dataloader(tag='valid')
        test_loader  = self._dataset.get_dataloader(tag='test')

        num_train_epochs = num_train_epochs or self._num_train_epochs
        self._net.set_train()
        best_rmse = 1e30
        for epoch in range(num_train_epochs):
            self._train_epoch(epoch, train_loader, net_mode=net_mode, weights=None)
            _, res = self.evaluate(epoch, test_loader,  tag='test', net_mode=net_mode)
            if res['rmse'][0] < best_rmse:
                best_rmse = res['rmse'][0]
                ms.save_checkpoint(self._net, f"./checkpoints/train_ckpt/ALLOT_best_{rank_id}.ckpt")

        ms.load_param_into_net(self._net, ms.load_checkpoint(f"./checkpoints/train_ckpt/ALLOT_best_{rank_id}.ckpt"))
        self.evaluate(num_train_epochs, test_loader, tag='test', net_mode=net_mode)
    
    def test(self, net_mode=Mode.ONE_PATH_FIXED):
        self.clear_records()
        test_loader  = self._dataset.get_dataloader(tag='test')
        self.evaluate(0, test_loader, tag='test', net_mode=net_mode)
        
    def evaluate(self, epoch, dataloader, tag, net_mode=Mode.ALL_PATHS, weights=None):
        speedometer = Speedometer(
            title=tag,
            epoch=epoch,
            metric_names=self._metric_names,
            metric_indexes=self._metric_indexes,
            print_frequency=self._print_frequency,
            batch_size=self._dataset._batch_size
        )
        self._net.set_train(False)
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(dataloader.create_tuple_iterator()):
            batch_x = ms.Tensor(batch_x.numpy(), dtype=ms.float32)
            batch_y = ms.Tensor(batch_y.numpy(), dtype=ms.float32)
            batch_x_mark = ms.Tensor(batch_x_mark.numpy(), dtype=ms.float32)
            
            preds = self._net(batch_x, batch_x_mark, 
                        attn_mask = None, adj_mats = ms.Tensor.from_numpy(self._dataset.adj_mats).float(), 
                        mode = net_mode, weights = weights)
            preds = self._dataset.scaler.inverse_transform(preds)
            # log metrics
            speedometer.update(preds, batch_y)
        self._net.set_train()

        print('-'*30)
        print('Epoch [{}] and Tag [{}] finished'.format(epoch, tag))
        print('-'*30)
        
        return speedometer.finish(), speedometer._metrics.get_value()

import time

class Speedometer:
    def __init__(self, title, epoch, metric_names, metric_indexes, print_frequency, batch_size):
        self._title = title
        self._epoch = epoch
        self._metric_names = metric_names
        self._metric_indexes = metric_indexes
        self._print_frequency = print_frequency
        self._batch_size = batch_size
        self.reset()

    def reset(self):
        self._metrics = Metric.Metrics(self._metric_names, self._metric_indexes)
        self._start = time.time()
        self._tic = time.time()
        self._counter = 0

    def update(self, preds, labels, step_size=1):
        self._metrics.update(preds, labels)
        self._counter += step_size
        if self._counter % self._print_frequency == 0:
            time_spent = time.time() - self._tic
            speed = float(self._print_frequency * self._batch_size) / time_spent
            out_str = [
                '[{}]'.format(self._title),
                'epoch[{}]'.format(self._epoch),
                'batch[{}]'.format(self._counter),
                'time: {:.2f}'.format(time_spent),
                'speed: {:.2f} samples/s'.format(speed),
                str(self._metrics)
            ]
            print('\t'.join(out_str))
            # logging.info('\t'.join(out_str))
            self._tic = time.time()

    def finish(self):
        out_str = [
            '[{}]'.format(self._title),
            'epoch[{}]'.format(self._epoch),
            'time: {:.2f}'.format((time.time() - self._start)),
            str(self._metrics)
        ]
        print('\t'.join(out_str))
        # logging.info('\t'.join(out_str))
        return self._metrics

