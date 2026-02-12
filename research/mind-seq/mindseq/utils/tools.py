import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore.ops import constexpr

import matplotlib.pyplot as plt

plt.switch_backend('agg')

@constexpr
def generate_tensor(t_shape):
    return ms.Tensor(np.ones(t_shape), ms.float32)
def get_attn_adj(scores, key=None, super_p=None, jat_sgn=False):
    A2 = ops.stop_gradient(scores)
    A2 = ops.where(ops.isnan(A2), ops.full_like(A2, 0), A2)
    bs, he, lq, lk = A2.shape
    len_key = ops.norm(key, dim = -1, keepdim=True)
    len_key[len_key < 1] = 1
    len_key = len_key.unsqueeze(-1)
    Ak2 = A2.swapaxes(-1, -2).unsqueeze(-1).matmul(A2.swapaxes(-1, -2).unsqueeze(-2)) / len_key

    if jat_sgn:
        sgn_mat = (A2 < 0).int()
        sgn_k_mat = sgn_mat.swapaxes(-1, -2).unsqueeze(-1).matmul(sgn_mat.swapaxes(-1, -2).unsqueeze(-2))
        Ak2[sgn_k_mat.bool()] = 0
    Ak2 = Ak2 / key.shape[-1]

    Ak_sorted_flat = np.sort(Ak2.asnumpy().flatten())
    super_p_part = int(len(Ak_sorted_flat) * super_p)
    super_p = Ak_sorted_flat[super_p_part].item()

    index_1 = Ak2 >= super_p
    Ak2 = ops.zeros_like(Ak2, dtype=ms.float32)
    Ak2[index_1] = 1
    A2 = ops.mean(Ak2[:,:,:,:,:], axis=2, keep_dims=False)
    diag_ = ops.arange(0, lk)
    A2[ops.arange(bs)[:, None, None],
        ops.arange(he)[None, :, None],
        ops.arange(lq)[None, None, :],
        diag_] = 1 #A_hat
    D = A2.sum(-1)
    D[D <= 1e-4] = 1e-4
    diag = ops.reciprocal(ops.sqrt(D)).unsqueeze(-1)
    A_wave = ops.mul(ops.mul(diag, A2), diag.swapaxes(-1, -2))
    return A_wave

def mask_fill(mask, data, num):
    select = ops.Select()
    # replace_tensor = generate_tensor(data.shape)
    # replace_tensor = ms.Tensor(np.ones(data.shape), ms.float32)
    # replace_tensor[:] = num
    replace_tensor = ops.fill(ms.float32, data.shape, num)
    return select(mask, replace_tensor, data.astype(ms.float32))

def adjust_learning_rate(optimizer, parameters, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        optimizer = ms.nn.Adam(parameters, learning_rate=lr)
        print('Updating learning rate to {}'.format(lr))
    return optimizer

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, rank_id=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.rank_id = rank_id

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        ms.save_checkpoint(model, path+'/'+f'checkpoint_{self.rank_id}.ckpt')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = ms.Tensor(self.mean, data.dtype) if "mindspore.common.tensor.Tensor" in str(type(data)) else self.mean
        std = ms.Tensor(self.std, data.dtype) if "mindspore.common.tensor.Tensor" in str(type(data)) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = ms.Tensor(self.mean, data.dtype) if "mindspore.common.tensor.Tensor" in str(type(data)) else self.mean
        std = ms.Tensor(self.std, data.dtype) if "mindspore.common.tensor.Tensor" in str(type(data)) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')