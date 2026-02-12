from ..data.data_factory import data_provider
from .Exp_basic import Exp_Basic
# from models import Informer, Autoformer, Transformer, Reformer
from .Autoformer import Autoformer
from ..utils.tools import EarlyStopping, adjust_learning_rate, visual
from ..utils.metrics import metric

# import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

import os
import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Autoformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Autoformer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
        }
        model = model_dict[self.args.model](self.args)
        return model

    def _get_data(self, flag):
        data_loader, data_set = data_provider(self.args, flag)
        return data_loader, data_set

    def _select_optimizer(self):
        model_optim = nn.Adam(self.model.trainable_params(), learning_rate=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
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
        train_data, train_source = self._get_data(flag='train')
        vali_data, vali_source = self._get_data(flag='val')
        test_data, test_source = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = train_data.get_dataset_size()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.set_train()

            def forward_fn(batch_x, batch_y, batch_x_mark, batch_y_mark):
                pred, true = self._process_one_batch(
                    train_source, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                return loss, true
        
            grad_fn = ops.value_and_grad(forward_fn, None, model_optim.parameters, has_aux=True)# change
            time_now = time.time()
            for i, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(train_data.create_tuple_iterator()):
                iter_count += 1
                (loss, _), grads = grad_fn(seq_x, seq_y, seq_x_mark, seq_y_mark)
                loss = ops.depend(loss, model_optim(grads))
                train_loss.append(loss.asnumpy())
                if (i+1) % 50==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.asnumpy()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_source, criterion)
            test_loss = self.vali(test_data, test_source, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        return 

    def test(self, setting, ckpt=None):
        test_data, test_source = self._get_data(flag = 'test')
        
        self.model.set_train(False)
        
        preds = []
        trues = []
        time_now = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_data.create_tuple_iterator()):
            pred, true = self._process_one_batch(
                test_source, batch_x, batch_y, batch_x_mark, batch_y_mark)
            if (i+1) % 100 == 0:
                print("iters: {0} time: {1}".format(i + 1, time.time() - time_now))
                time_now = time.time()
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
        dec_inp = ops.Zeros()((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]), ms.float32)
        dec_inp = cast(ops.concat([batch_y[:,:self.args.label_len,:], dec_inp], axis=1), ms.float32)
        # encoder - decoder
        if self.args.output_attention:
            res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, A = res[0], res[1]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:]
        if self.args.output_attention:
            return outputs, batch_y, A
        return outputs, batch_y