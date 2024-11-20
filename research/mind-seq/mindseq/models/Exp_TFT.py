from .Exp_basic import Exp_Basic
from .TFT import TemporalFusionTransformer, CONFIGS

from ..data.data_utils import load_dataset
from ..utils.criterions import QuantileLoss
import pandas as pd

from mindspore.communication import get_group_size, get_rank, init
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import os
import time

FEAT_NAMES = ['s_cat' , 's_cont' , 'k_cat' , 'k_cont' , 'o_cat' , 'o_cont' , 'target', 'id']
import argparse
import time
import os
import pickle
import json
import mindspore as ms
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

def validate(config, model, criterion, dataloader, global_step, tp="val"):
    if tp == "val":
        if not hasattr(validate, 'best_valid_loss'):
            validate.best_valid_loss = float('inf')
        if not hasattr(validate, 'early_stop_c'):
            validate.early_stop_c = 0
    
    model.set_train(False)
    losses = []
    for i, item in enumerate(dataloader.create_tuple_iterator()):
        batch = {key: tensor if tensor.numel() > tensor.shape[0] else None for key, tensor in zip(FEAT_NAMES, item)}
        predictions = model(batch)
        targets = batch['target'][:,config.encoder_length:,:]
        p_losses = criterion(predictions, targets)
        bs = next(t for t in batch.values() if t is not None).shape[0]
        losses.append((p_losses, bs))

    p_losses = sum([l[0]*l[1] for l in losses])/sum([l[1] for l in losses]) #takes into accunt that the last batch is not full

    log_dict = {'P10':p_losses[0].numpy().item(), 'P50':p_losses[1].numpy().item(), 'P90':p_losses[2].numpy().item(), 'loss': p_losses.sum().numpy().item()}
    if tp == "val":
        if log_dict['loss'] < validate.best_valid_loss:
            validate.best_valid_loss = log_dict['loss']
            validate.early_stop_c = 0
            validate.conv_step = global_step
        else:
            validate.early_stop_c += 1
        
    log_dict = {tp+"_"+k:v for k,v in log_dict.items()}
    print("step", global_step, "log_dict", log_dict)

class Exp_TFT(Exp_Basic):
    def __init__(self, args):
        super(Exp_TFT, self).__init__(args)
        self.args.distributed_world_size = 1

        self.config = CONFIGS[self.args.dataset]()
        if self.args.overwrite_config:
            self.config.__dict__.update(json.loads(self.args.overwrite_config))
        
        self.train_loader, self.valid_loader, self.test_loader = load_dataset(self.args, self.config)
    
        print("Run dummy iteration to initialize lazy modules")
        timer = time.time()
        for dummy_item in self.train_loader.create_tuple_iterator():
            dummy_batch = {key: tensor if tensor.numel() > tensor.shape[0] else None for key, tensor in zip(FEAT_NAMES, dummy_item)}
            self.model(dummy_batch)
            print("dummy", time.time() - timer)
            break

        self.criterion = QuantileLoss(self.config)
        self.optimizer = ms.nn.Adam(self.model.trainable_params(), learning_rate=self.args.lr)

        self.global_step = 0
        
        self.tgt_scalers = pickle.load(open(os.path.join(self.args.data_path, 'tgt_scalers.bin'), 'rb'))
        self.cat_encodings = pickle.load(open(os.path.join(self.args.data_path,'cat_encodings.bin'), 'rb'))
        
    def forward_fn(self,batch):
        forecast = self.model(batch)
        targets = batch['target'][:,self.config.encoder_length:,:]
        p_losses = self.criterion(forecast, targets)
        return p_losses.sum(), forecast, p_losses
    def _build_model(self):
        config = CONFIGS[self.args.dataset]()
        return TemporalFusionTransformer(config)
    def _get_model(self):
        return self.model
    def _unscale_per_id(self, config, values, ids, scalers):
        num_horizons = config.example_length - config.encoder_length + 1
        flat_values = pd.DataFrame(
                values,
                columns=[f't{j}' for j in range(num_horizons - values.shape[1], num_horizons)]
                )
        flat_values['id'] = ids
        df_list = []
        for idx, group in flat_values.groupby('id'):
            scaler = scalers[idx]
            group_copy = group.copy()
            for col in group_copy.columns:
                if not 'id' in col:
                    _col = np.expand_dims(group_copy[col].values, -1)
                    _t_col = scaler.inverse_transform(_col)[:,-1]
                    group_copy[col] = _t_col
            df_list.append(group_copy)
        flat_values = pd.concat(df_list, axis=0)

        flat_values = flat_values[[col for col in flat_values if not 'id' in col]]
        return flat_values.values
    
    
    def train(self):
        # train_loader, valid_loader, test_loader = load_dataset(self.args, self.config)
        best_50 = 1e30
        best_q = None
        print("Train!")
        if self.args.distribute:
            mean = ms.context.get_auto_parallel_context("gradients_mean")
            degree = ms.context.get_auto_parallel_context("device_num")
            grad_reducer = ms.nn.DistributedGradReducer(self.optimizer.parameters, mean, degree)

        ms.load_param_into_net(self.model, ms.load_checkpoint("./checkpoints/test_ckpt/TFT_pre.ckpt"))
        rank_id = 0 if self.args.rank_id is None else self.args.rank_id
        for epoch in range(self.args.epochs):
            self.model.set_train()
            timer = time.time()
            print("len_train = {}".format(len(self.train_loader)))
            for i, item in enumerate(self.train_loader.create_tuple_iterator()):
                batch = {key: tensor if tensor.numel() > tensor.shape[0] else None for key, tensor in zip(FEAT_NAMES, item)}
                grad_fn = ms.ops.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
                (loss, _, p_losses), grads = grad_fn(batch)
                if self.args.distribute:
                    grads = grad_reducer(grads)
                loss = ms.ops.depend(loss, self.optimizer(grads))
                log_dict = {'train_P10':p_losses[0].numpy().item(), 'train_P50':p_losses[1].numpy().item(), 'train_P90':p_losses[2].numpy().item(), 'sum': loss.numpy().item()}
                if self.global_step % 10 == 0:
                    print("time", time.time() - timer)
                    print("step", self.global_step, "log_dict", log_dict)
                    timer = time.time()
                self.global_step += 1
            print("Train_Loader OK!")
            # validate(self.config, self.model, self.criterion, self.valid_loader, self.global_step, tp="val")
            # validate(self.config, self.model, self.criterion, self.test_loader, self.global_step, tp="test")

            # if validate.early_stop_c >= self.args.early_stopping:
            #     print('Early stopping')
            #     break
            
            self.model.set_train(False)
            unscaled_predictions, unscaled_targets = self.predict(self.test_loader, self.tgt_scalers, self.cat_encodings)
            unscaled_predictions = ms.Tensor(unscaled_predictions)
            unscaled_targets = ms.Tensor(unscaled_targets)
            
            losses = self.criterion(unscaled_predictions, unscaled_targets)
            normalizer = unscaled_targets.abs().mean()
            quantiles = 2 * losses / normalizer
            
            test_50 = quantiles[1].numpy().item()
            quantiles = {'test_p10': quantiles[0].numpy().item(), 'test_p50': quantiles[1].numpy().item(), 'test_p90': quantiles[2].numpy().item(), 'sum':sum(quantiles).numpy().item()}
            finish_log = {**quantiles}
            print(finish_log)
            if test_50 < best_50:
                best_50 = test_50
                best_q = quantiles
                ms.save_checkpoint(self.model, f"./checkpoints/train_ckpt/TFT_best_{rank_id}.ckpt")
        print(best_q)

    def test(self):
        rank_id = 0 if self.args.rank_id is None else self.args.rank_id
        print(f"Test Rank {rank_id}")
        if not self.args.do_train:
            print(f"Load From {self.args.ckpt_path}")
            ms.load_param_into_net(self.model, ms.load_checkpoint(self.args.ckpt_path))
        else:
            ms.load_param_into_net(self.model, ms.load_checkpoint(f"./checkpoints/train_ckpt/TFT_best_{rank_id}.ckpt"))
        self.model.set_train(False)
        unscaled_predictions, unscaled_targets = self.predict(self.test_loader, self.tgt_scalers, self.cat_encodings)
        unscaled_predictions = ms.Tensor(unscaled_predictions)
        unscaled_targets = ms.Tensor(unscaled_targets)
        
        losses = self.criterion(unscaled_predictions, unscaled_targets)
        normalizer = unscaled_targets.abs().mean()
        quantiles = 2 * losses / normalizer
        quantiles = {'test_p10': quantiles[0].numpy().item(), 'test_p50': quantiles[1].numpy().item(), 'test_p90': quantiles[2].numpy().item(), 'sum':sum(quantiles).numpy().item()}
        print(quantiles)

    def predict(self, data_loader, scalers, cat_encodings, extend_targets=False):
        predictions = []
        targets = []
        ids = []
        
        self.model.set_train(False)
        timer = time.time()
        for i, item in enumerate(data_loader.create_tuple_iterator()):
            batch = {key: tensor if tensor.numel() > tensor.shape[0] else None for key, tensor in zip(FEAT_NAMES, item)}
            ids.append(batch['id'][:,0,:])
            targets.append(batch['target'])
            predictions.append(self.model(batch).float())
            if (i+1) % 10 == 0:
                print("iters: {0} time: {1}".format(i + 1, time.time() - timer))
                timer = time.time()
        targets = ms.ops.cat(targets, axis=0).asnumpy()
        if not extend_targets:
            targets = targets[:,self.config.encoder_length:,:] 
        predictions = ms.ops.cat(predictions, axis=0).asnumpy()
        ids = ms.ops.cat(ids, axis=0).asnumpy()

        unscaled_predictions = np.stack(
                [self._unscale_per_id(self.config, predictions[:,:,i], ids, scalers) for i in range(len(self.config.quantiles))], 
                axis=-1)
        unscaled_targets = np.expand_dims(self._unscale_per_id(self.config, targets[:,:,0], ids, scalers), axis=-1)
        return unscaled_predictions, unscaled_targets