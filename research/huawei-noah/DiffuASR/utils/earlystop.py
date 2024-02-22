'''
This code based on https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
The origin code conduct early stopping according to validation loss,
I alter it to early stop according to validation performance.
'''


import numpy as np
import mindspore
import os


class EarlyStopping():
    """Early stops the training if validation performance doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='./checkpoint/', trace_func=print, model='checkpoint'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        if not os.path.exists(path):
            os.makedirs(path)

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None  # record the best score
        self.best_epoch = 0 # record the best epoch
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(path, "pytorch_model.bin")
        self.trace_func = trace_func


    def __call__(self, indicator, epoch, model):

        score = indicator

        if self.best_score is None: # for the first epoch
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'The best score is ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        mindspore.save(model.state_dict(), self.path)
        #self.val_loss_min = val_loss



class EarlyStoppingNew():
    """Early stops the training if validation performance doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='./checkpoint/', trace_func=print, model='checkpoint'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        if not os.path.exists(path):
            os.makedirs(path)

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None  # record the best score
        self.best_epoch = 0 # record the best epoch
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(path, "pytorch_model.bin")
        self.trace_func = trace_func


    def __call__(self, indicator, epoch, model, optimizer=None, scheduler=None):

        score = indicator

        if self.best_score is None: # for the first epoch
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model, optimizer, scheduler, epoch)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model, optimizer, scheduler, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'The best score is ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        mindspore.save_checkpoint(model, self.path)
        