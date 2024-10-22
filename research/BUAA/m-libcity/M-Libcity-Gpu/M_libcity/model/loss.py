import mindspore
from mindspore import numpy as mnp
import numpy as np
from mindspore.nn import LossBase
from sklearn.metrics import r2_score, explained_variance_score
from functools import partial



class LibCityLoss(LossBase):
    def __init__(self,loss_name):
        super(LibCityLoss, self).__init__()
        self.loss_name=loss_name
        if loss_name.lower() == 'mae':
            self.lf = masked_mae_m
        elif loss_name.lower() == 'mse':
            self.lf = masked_mse_m
        elif loss_name.lower() == 'rmse':
            self.lf = masked_rmse_m
        elif loss_name.lower() == 'mape':
            self.lf = masked_mape_m
        elif loss_name.lower() == 'logcosh':
            self.lf = log_cosh_loss
        elif loss_name.lower() == 'huber':
            self.lf = huber_loss
        elif loss_name.lower() == 'quantile':
            self.lf = quantile_loss
        elif loss_name.lower() == 'masked_mae':
            self.lf = partial(masked_mae_m, null_val=0)
        elif loss_name.lower() == 'masked_mse':
            self.lf = partial(masked_mse_m, null_val=0)
        elif loss_name.lower() == 'masked_rmse':
            self.lf = partial(masked_rmse_m, null_val=0)
        elif loss_name.lower() == 'masked_mape':
            self.lf = partial(masked_mape_m, null_val=0)
        elif loss_name.lower() == 'r2':
            self.lf = r2_score_m
        elif loss_name.lower() == 'evar':
            self.lf = explained_variance_score_m
        else:
            self.lf = masked_mae_m
    def construct(self, logits, labels):
        if self.loss_name=="mask_mape":
            return self.lf(logits,labels,eps=7e-6)
        else:
            return self.lf(logits,labels)



def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).astype(mindspore.float32)
    mask /= mask.mean()
    loss = mnp.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans:
    # https://discuss.pymnp.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def masked_mae_m(preds, labels, null_val=np.nan):
    labels[mnp.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~mnp.isnan(labels)
    else:
        null_val=mindspore.Tensor(null_val)
        mask = mnp.not_equal(labels,null_val) # not equal to null_val
    mask = mask.astype(mindspore.float16)
    mask /= mnp.mean(mask)
    mask = mnp.where(mnp.isnan(mask), mnp.zeros_like(mask), mask)
    loss = mnp.abs(mindspore.ops.sub(preds, labels))
    loss = loss * mask
    loss = mnp.where(mnp.isnan(loss), mnp.zeros_like(loss), loss)
    return mnp.mean(loss)



def log_cosh_loss(preds, labels):
    loss = mnp.log(mnp.cosh(preds - labels))
    return mnp.mean(loss)


def huber_loss(preds, labels, delta=1.0):
    residual = mnp.abs(preds - labels)
    condition = mindspore.ops.le(residual, delta)
    small_res = 0.5 * mnp.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return mnp.mean(mnp.where(condition, small_res, large_res))



def quantile_loss(preds, labels, delta=0.25):
    condition = mnp.ge(labels, preds)
    large_res = delta * (labels - preds)
    small_res = (1 - delta) * (preds - labels)
    return mnp.mean(mnp.where(condition, large_res, small_res))


def masked_mape_m(preds, labels, null_val=mnp.nan, eps=0):
    labels[mnp.abs(labels) < 1e-4] = 0
    if mnp.isnan(null_val) and eps != 0:
        loss = mnp.abs((preds - labels) / (labels + eps))
        loss[mnp.isinf(loss)] = 1
        loss[mnp.isnan(loss)] = 0
        return mnp.mean(loss)
    if mnp.isnan(null_val):
        mask = ~mnp.isnan(labels)
    else:
        null_val=mindspore.Tensor(null_val)
        mask = mnp.not_equal(labels,null_val)
    mask = mask.astype(mindspore.float32)
    mask /= mnp.mean(mask)
    mask = mnp.where(mnp.isnan(mask), mnp.zeros_like(mask), mask)
    loss = mnp.abs((preds - labels)/labels)
    loss = loss * mask
    loss = mnp.where(mnp.isnan(loss), mnp.zeros_like(loss), loss)
    loss = mnp.where(mnp.isinf(loss), mnp.zeros_like(loss), loss)
    return mnp.mean(loss)


def masked_mse_m(preds, labels, null_val=np.nan):

    labels[mnp.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~mnp.isnan(labels)
    else:
        null_val=mindspore.Tensor(null_val)
        mask = mnp.not_equal(labels,null_val)
    mask = mask.astype(mindspore.float32)
    mask /= mnp.mean(mask)
    mask = mnp.where(mnp.isnan(mask), mnp.zeros_like(mask), mask)
    mask = mnp.where(mnp.isinf(mask), mnp.zeros_like(mask), mask)
    loss = mindspore.ops.sub(preds, labels)
    loss = mnp.square(loss)
    loss = loss * mask
    loss = mnp.where(mnp.isnan(loss), mnp.zeros_like(loss), loss)
    loss = mnp.where(mnp.isinf(loss), mnp.zeros_like(loss), loss)
    return mnp.mean(loss)


def masked_rmse_m(preds, labels, null_val=np.nan):
    labels[mnp.abs(labels) < 1e-4] = 0
    return mnp.sqrt(masked_mse_m(preds=preds, labels=labels,
                                       null_val=null_val))


def r2_score_m(preds, labels):
    preds = preds.flatten().asnumpy()
    labels = labels.flatten().asnumpy()
    return r2_score(labels, preds)


def explained_variance_score_m(preds, labels):
    preds = preds.flatten().asnumpy()
    labels = labels.flatten().asnumpy()
    return explained_variance_score(labels, preds)


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels,
                   null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(
            preds, labels).astype('float32'), labels))
        print(mape[2][935])
        print(mape[6][183])
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def r2_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return r2_score(labels, preds)


def explained_variance_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return explained_variance_score(labels, preds)