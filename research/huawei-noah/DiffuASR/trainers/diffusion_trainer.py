# -*- encoding: utf-8 -*-
# here put the import lib
import os
import numpy as np
import mindspore
import mindspore.ops as ops
from tqdm import tqdm, trange
from trainers.trainer import Trainer
from utils.utils import load_pretrained_model, seq_acc
from diffusion.diffusion import DiffusionModel, DiffusionModel_CG, DiffusionModel_CF
from diffusion.ema import EMAHelper



class DiffusionTrainer(Trainer):

    def __init__(self, args, logger, generator):

        super().__init__(args, logger, generator)

        if args.pretrain_item:

            self._load_rec_model(args.rec_path)

        if args.freeze_item:

            self._freeze_item()

    
    def _create_model(self):
        
        if self.args.model_name == 'diffusion':
            if self.args.guide_model == "none":
                self.model = DiffusionModel(self.user_num, self.item_num, self.args)
            elif self.args.guide_model == "cg":
                self.model = DiffusionModel_CG(self.user_num, self.item_num, self.args)
            elif self.args.guide_model == "cf":
                self.model = DiffusionModel_CF(self.user_num, self.item_num, self.args)
        else:
            raise NotImplementedError

    
    def load_model(self):

        self.model = load_pretrained_model(self.args.pretrain_dir, self.model, self.logger)

    
    def _load_rec_model(self, rec_path):

        self.logger.info("Loading recommendation model ... ")
        checkpoint_path = os.path.join(rec_path, 'pytorch_model.bin')

        model_dict = self.model.state_dict()
        pretrained_dict = mindspore.load_checkpoint(checkpoint_path)

        # filter out required parameters
        #required_params = ['item_emb']
        new_dict = {k: v for k, v in pretrained_dict.items() if "item_emb" in k}
        model_dict.update(new_dict)
        self.logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        mindspore.load_param_into_net(self.model, model_dict)

    
    def _freeze_item(self):

        for name, param in self.model.named_parameters():
    
            if 'item_emb' in name:

                param.requires_grad = False


    def train(self):

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Batch size = %d", self.args.train_batch_size)

        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch"):

            avg_loss = self._train_one_epoch(epoch)

            if (epoch+1)%100 == 0:

                self.eval(epoch)

            self.stopper(-avg_loss, epoch, model_to_save, self.optimizer)

            if self.stopper.early_stop:
                break
        
        best_epoch = self.stopper.best_epoch
        self.logger.info('')
        self.logger.info('The best epoch is %d' % best_epoch)


    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        self.model.set_train(True)
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')
        loss_list = []

        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Epoch: %d Train **********" % epoch)

        def forward_fn(batch):
            seq, positions, diff_seq = batch
            seq, positions, diff_seq = seq.long(), positions.long(), diff_seq.long()
            loss = self.model(diff_seq, seq)
            return loss, seq
        grad_fn = mindspore.ops.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)

        for batch in prog_iter:

            (loss, _), grads = grad_fn(batch)
            loss_list.append(loss.item())
            self.optimizer(grads)

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

        self.logger.info("The loss value %.5f" % np.mean(loss_list))

        return np.mean(loss_list)


    def augment(self):

        self.model.set_train(False)
        aug_data = []
        aug_loader = self.generator.make_augmentloader()

        for batch in tqdm(aug_loader):

            seq, positions = batch
            seq, positions = seq.long(), positions.long()
            item_indicies = ops.arange(1, self.item_num+1)    # (1, item_num) to get the item embedding matrix
            item_indicies = item_indicies.tile((seq.shape[0], 1))   # (bs, item_num)
            item_indicies = item_indicies.long()
            per_aug_data = None

            logits = self.model.predict(seq, item_indicies)

            for i in range(self.args.aug_num):
                
                logit = logits[i]
                aug_item = ops.argsort(logit, descending=True)[:, 0]   # return the index of max score
                aug_item = aug_item + 1
                aug_item = aug_item.unsqueeze(1)    # (bs, 1)
                if per_aug_data is None:
                    per_aug_data = aug_item
                else:
                    per_aug_data = ops.cat([per_aug_data, aug_item], axis=1)  # [..., n-3, n-2, n-1]

            aug_data.append(per_aug_data)

        aug_data = ops.cat(aug_data, axis=0)

        aug_data = aug_data.numpy()
        self.generator.save_aug(aug_data)
    
        

