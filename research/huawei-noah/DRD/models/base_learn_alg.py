import sys
import os
import random
import time
# import torch
import pandas as pd
import numpy as np
# import torch.nn as nn
# from torch.nn import Module
# from torch.nn import init
# from torch.optim import Adam,Adadelta,RMSprop,SGD
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.nn import BCELoss,BCEWithLogitsLoss
# from torch.utils.data import DataLoader, Dataset
import mindspore
from mindspore import nn
from mindspore.common.initializer import Normal,XavierNormal, Initializer,initializer
from utils.set_seed import setup_seed
from utils.data_wrapper import Wrap_Dataset_point1
from utils.early_stop import EarlyStopping
from utils.arg_parser import config_param_parser
from utils.pairwise_trans import get_pair, get_pair_fullinfo
from utils.trans_format import format_trans
# from utils.get_sparse import get_sparse_feature
from models.evaluate import Test_Evaluator
from models.toy_model import rank_model
# from models.loss_func import BPRLoss, HingeLoss


class Learning_Alg(object):
    
    def __init__(self, args):
        self.fin = args.fin
        self.fout = args.fout
        self.use_cuda = args.use_cuda
        self.train_alg = args.train_alg
        self.pairwise = args.pairwise # add this arg
        self.eval_positions = args.eval_positions
        self.topK = args.topK
        self.randseed = args.randseed
        self.continue_tag = args.continue_tag
        self.eval_tag = args.eval_tag
        self.min_alpha = args.min_alpha
        self.alpha = args.alpha
        self.beta = args.beta
        self.init_prob = args.init_prob
        self.eta = args.eta
        self.clip = args.clip
        self.optimizer = args.optimizer
        self.schedul_mode = args.schedul_mode
        self.session_num = args.session_num
        self.lr = args.lr
        self.drop_out = args.drop_out
        self.weight_decay = args.weight_decay
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.patience = args.patience
        self.hidden_size = args.hidden_size     
        
        
    def train(self):
        self.test_tool, self.in_size = self._load_test_and_get_tool()
        input_train_loader = self._load_and_wrap_train()
        self.model, self.optim, self.scheduler, self.early_stopping = self._init_train_env()
        self._train_iteration(input_train_loader)
        self._test_and_save()
    

    def _test_and_save(self):
        # load best model
        # self.model.load_state_dict(torch.load('{}_checkpoint.pt'.format(self.fout)))
        param_dict = mindspore.load_checkpoint('{}_checkpoint.ckpt'.format(self.fout))
        param_not_load = mindspore.load_param_into_net(self.model, param_dict)
        print(param_not_load)
        
        # evaluate model on test set
        test_result = self.test_tool.evaluate(self.model)

        # save model and result as given path
        #print('test_loss:', test_loss)
        # torch.save(self.model.state_dict(), '{}_model.pt'.format(self.fout))
        mindspore.save_checkpoint(self.model, '{}_model.ckpt'.format(self.fout))
        test_result.to_json('{}_result.json'.format(self.fout), indent=4)


    def _train_iteration(self, input_train_loader):
        # strat training
        dur = []
        for epoch in range(self.epoch):
            if epoch >= 3:
                t0 = time.time()
            
            loss_log = []
            self.model.train()

            for _id, batch in enumerate(input_train_loader):
                train_loss = self._train_one_batch(batch)
                loss_log.append(train_loss.item())

            test_result = self.test_tool.evaluate(self.model)
            
            ndcg_tst = test_result['NDCG'][self.topK]
            mrr_tst = test_result['MRR'][self.topK]
            precision_tst = test_result['Precision'][self.topK]
            map_tst = test_result['MAP'][self.topK]
            ndcg_full_tst = test_result['NDCG_full'][self.topK]

            # torch.save(self.model.state_dict(), '{}_checkpoint.pt'.format(self.fout))
            mindspore.save_checkpoint(self.model, '{}_checkpoint.ckpt'.format(self.fout))
            
            if epoch >= 3:
                dur.append(time.time() - t0)
            """
            for name, parms in self.model.named_parameters():
	                print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
            """
            # print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Val_Loss {:.4f} | Val_NDCG@10 {:.4f} | "
            #         "Val_MRR@10 {:.4f}| Val_Precision@10 {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),val_loss,
            #                                         ndcg_val, mrr_val, precision_val))
            print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Test_NDCG@10 {:.4f} | "
                    "Test_MRR@10 {:.4f}| Test_Precision@10 {:.4f} | Test_MAP@10 {:.4f} | Test_NDCG_full {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),ndcg_tst,
                                                    mrr_tst, precision_tst, map_tst, ndcg_full_tst))

   
    def _train_one_batch(self, batch):
        pass
        # self.optim.zero_grad()
        # if self.pairwise:
        #     BPR_lossfunc = BPRLoss(weight=batch[2])
        #     output_posi = self.model(batch[0])
        #     output_nega = self.model(batch[1])
        #     output_posi = output_posi.view(batch[0].size(0))
        #     output_nega = output_nega.view(batch[1].size(0))
        #     train_loss = BPR_lossfunc(output_posi, output_nega)
        # else:
        #     BCE_lossfunc = BCELoss()
        #     output = self.model(batch[0])
        #     output = output.view(batch[0].size(0))
        #     train_loss = BCE_lossfunc(output, batch[1])
        # train_loss.backward()
        # self.optim.step()
        # return train_loss


    def _init_train_env(self):
        model = rank_model(self.in_size, self.hidden_size, self.drop_out)
        # model.weight_init()
        for name, param in model.parameters_and_names():
            if 'weight' in name:
                param.set_data(initializer(XavierNormal(), param.shape, param.dtype))
            if 'bias' in name:
                param.set_data(initializer('zeros', param.shape, param.dtype))
                
        if self.use_cuda:
            #model = nn.DataParallel(model)
            model = model.cuda()
        
        if self.optimizer == 'adam':
            optim = nn.Adam(model.parameters(), learning_rate=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optim = nn.SGD(model.parameters(), learning_rate=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer== 'adadelta':
            optim = nn.Adadelta(model.parameters(), learning_rate=self.lr, weight_decay=self.weight_decay)
        # scheduler = ReduceLROnPlateau(optim, 
        #                               patience=10, 
        #                               mode=self.schedul_mode,
        #                               threshold=1e-6,
        #                               verbose=True)
        scheduler = 0
        early_stopping = EarlyStopping(self.fout, self.patience, verbose=True) 
        return model, optim, scheduler, early_stopping


    def _load_test_and_get_tool(self):
        test_dat = pd.read_json(self.fin + 'json_file/Test.json')
        test_dat = format_trans(test_dat,mode='test')
        test_tool = Test_Evaluator(test_dat, 
                                   self.eval_positions, 
                                   use_cuda=self.use_cuda)
        in_size = len(test_dat['feature'][0])
        return test_tool, in_size


    def _load_vali_and_get_tool(self):
        pass


    def _load_and_wrap_train(self):
        pass


if __name__=="__main__":
    pass