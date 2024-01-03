"""
-*- coding: utf-8 -*-
@Time    : 9/12/2023 10:40 pm
@Author  : Xiaopeng Li
@File    : callback.py

"""
import copy


class EarlyStopper(object):
    """Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience (int): How long to wait after the last time validation AUC improved.
    """

    def __init__(self, patience):
        self.patience = patience
        self.trial_counter = 0
        self.best_auc = 0
        self.best_weights = None

    def stop_training(self, val_auc, weights):
        """Whether to stop training.

        Args:
            val_auc (float): AUC score on validation data.
            weights (Parameter or Tensor): The weights of the model.
        """
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.trial_counter = 0
            # Ensure correct deep copy method for MindSpore tensors or parameters
            self.best_weights = copy.deepcopy(weights)
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True
