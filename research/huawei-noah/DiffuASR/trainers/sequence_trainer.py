# -*- encoding: utf-8 -*-
# here put the import lib
import os
import time
import mindspore
import mindspore.ops as ops
import numpy as np
from tqdm import tqdm
from trainers.trainer import Trainer
from utils.utils import metric_report, metric_len_report, record_csv


class SeqTrainer(Trainer):

    def __init__(self, args, logger, generator):

        super().__init__(args, logger, generator)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.set_train(True)
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        def forward_fn(batch):
            seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            loss = self.model(seq, pos, neg, positions)
            return loss, seq
        grad_fn = mindspore.ops.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)
        
        for batch in prog_iter:

            train_start = time.time()
            (loss, _), grads = grad_fn(batch)
            self.optimizer(grads)

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            train_end = time.time()
            train_time.append(train_end-train_start)



    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = mindspore.load_checkpoint(os.path.join(self.args.output_dir, 'pytorch_model.bin.ckpt'))
            mindspore.load_param_into_net(self.model, model_state_dict)
            #self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.set_train(False)
        pred_rank = mindspore.Tensor([1]).long()#.to(self.device)
        seq_len = mindspore.Tensor([1]).long()#.to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            #batch = tuple(t.to(self.device) for t in batch)
            seq, pos, neg, positions = batch
            seq_len = ops.cat([seq_len, ops.sum((seq>0), dim=1)])
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            
            pred_logits = -self.model.predict(seq, ops.cat([pos, neg], axis=-1), positions)

            per_pred_rank = ops.argsort(ops.argsort(pred_logits))[:, 0]
            pred_rank = ops.cat([pred_rank, per_pred_rank.long()])

        self.logger.info('')
        res_dict = metric_report(pred_rank.numpy())
        res_len_dict = metric_len_report(pred_rank.numpy(), seq_len.numpy(), aug_len=self.args.aug_seq_len)
        
        for k, v in res_dict.items():
            self.logger.info('%s: %.5f' % (k, v))
        for k, v in res_len_dict.items():
            self.logger.info('%s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict}

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict

