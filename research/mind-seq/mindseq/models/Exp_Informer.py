from ..data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from .Exp_basic import Exp_Basic
from mindspore.dataset import GeneratorDataset
from .Informer import Informer

from ..utils.tools import EarlyStopping
from ..utils.metrics import metric

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'Informer':Informer,
        }
        if self.args.model=='Informer':
            e_layers = self.args.e_layers if self.args.model=='Informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.args,
            )
        
        return model
    
    def _get_model(self):
        return self.model

    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'weather': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        source_data = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        if flag == 'test':
            data_set = GeneratorDataset(source=source_data, column_names=["seq_x", "seq_y", "seq_x_mark", "seq_y_mark"])
        else:
            data_set = GeneratorDataset(source=source_data, column_names=["seq_x", "seq_y", "seq_x_mark", "seq_y_mark"], num_shards=args.device_num, shard_id = args.rank_id)
        print(flag, data_set.get_dataset_size())
        if shuffle_flag:
            data_set = data_set.shuffle(data_set.get_dataset_size())
        data_set = data_set.batch(batch_size = batch_size, drop_remainder=drop_last)
        return data_set, source_data

    def _select_optimizer(self):
        model_optim = nn.Adam(self.model.trainable_params(), learning_rate=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_source, criterion):
        self.model.set_train(False)
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_data.create_tuple_iterator()):
            pred, true = self._process_one_batch(
                vali_source, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred, true).asnumpy()
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.set_train()
        return total_loss

    def train(self, setting):
        train_data, train_source = self._get_data(flag = 'train')
        vali_data, vali_source = self._get_data(flag = 'val')
        test_data, test_source = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = train_data.get_dataset_size()
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        def forward_fn(batch_x, batch_y, batch_x_mark, batch_y_mark, padding, label_len, pred_len):
            cast = ops.Cast()
            batch_x = cast(batch_x, ms.float32)
            batch_y = cast(batch_y, ms.float32)

            batch_x_mark = cast(batch_x_mark, ms.float32)
            batch_y_mark = cast(batch_y_mark, ms.float32)

            dec_inp = ops.Zeros()((batch_y.shape[0], pred_len, batch_y.shape[-1]), ms.float32)
            if padding==1:
                dec_inp = ops.Ones()((batch_y.shape[0], pred_len, batch_y.shape[-1]), ms.float32)
            dec_inp = cast(ops.concat([batch_y[:,:label_len,:], dec_inp], axis=1), ms.float32)

            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            batch_y = batch_y[:,-pred_len:,0:]
            loss = criterion(outputs, batch_y)
            return loss, outputs
        rank_id = 0 if self.args.rank_id is None else self.args.rank_id
        time_now = time.time()
        self.model.set_train()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, rank_id=rank_id)
        if self.args.distribute:
            mean = ms.context.get_auto_parallel_context("gradients_mean")
            degree = ms.context.get_auto_parallel_context("device_num")
            grad_reducer = ms.nn.DistributedGradReducer(model_optim.parameters, mean, degree)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_data.create_tuple_iterator()):
                iter_count += 1
                grad_fn = ms.ops.value_and_grad(forward_fn, None, model_optim.parameters, has_aux=True)
                (loss, _), grads = grad_fn(batch_x, batch_y, batch_x_mark, batch_y_mark, 
                                            self.args.padding, self.args.label_len, self.args.pred_len)
                if self.args.distribute:
                    grads = grad_reducer(grads)
                loss = ms.ops.depend(loss, model_optim(grads))
                train_loss.append(loss.asnumpy().item())

                if (i+1) % 50 ==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.asnumpy().item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
        
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_source, criterion)
            test_loss = self.vali(test_data, test_source, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        best_model_path = path+'/'+f'checkpoint_{rank_id}.ckpt'
        ms.load_param_into_net(self.model, ms.load_checkpoint(best_model_path))

    def test(self, setting, ckpt = None):
        test_data, test_source = self._get_data(flag = 'test')

        self.model.set_train(False)
        
        preds = []
        trues = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_data.create_tuple_iterator()):
            pred, true = self._process_one_batch(
                test_source, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.numpy())
            trues.append(true.numpy())
        self.model.set_train()
        preds = np.array(preds)
        trues = np.array(trues)
        
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape([-1, preds.shape[-2], preds.shape[-1]])
        trues = trues.reshape([-1, trues.shape[-2], trues.shape[-1]])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        cast = ops.Cast()
        batch_x = cast(batch_x, ms.float32)
        batch_y = cast(batch_y, ms.float32)

        batch_x_mark = cast(batch_x_mark, ms.float32)
        batch_y_mark = cast(batch_y_mark, ms.float32)

        # decoder input
        if self.args.padding==0:
            dec_inp = ops.Zeros()((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]), ms.float32)
        elif self.args.padding==1:
            dec_inp = ops.Ones()((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]), ms.float32)
        dec_inp = cast(ops.concat([batch_y[:,:self.args.label_len,:], dec_inp], axis=1), ms.float32)
        # encoder - decoder
        if self.args.output_attention:
            res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, A = res[0], res[1]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:]
        return outputs, batch_y
