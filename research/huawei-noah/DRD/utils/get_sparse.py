import torch
import numpy as np


def pre_trans_row(row):
    num_row = [[float(fe[0]), float(fe[1])] for fe in row]
    return num_row

def get_sparse_feature_forOnerow(row):
    idx = np.array(row, dtype=int)[:,0] - 1      
    val = np.array(row, dtype=float)[:,1]
    if 699 not in idx:
        idx = np.append(idx, 699) 
        val = np.append(val, 0.)
    sparse_fe = torch.sparse.FloatTensor(torch.LongTensor([idx]), torch.FloatTensor(val))
    return sparse_fe

def get_sparse_feature(dat):
    row_idx_ls = []
    col_idx_ls = []
    val_ls = []
    # dat = dat.apply(pre_trans_row)
    for i in range(len(dat)):
        row_idx_ls.extend([i]*len(dat[i])) 
        col_idx_ls.extend(np.array(dat[i], dtype=int)[:,0] - 1)
        val_ls.extend(np.array(dat[i], dtype=float)[:,1])
    sparse_ts = torch.sparse.FloatTensor(torch.LongTensor([row_idx_ls, col_idx_ls]), torch.FloatTensor(val_ls))
    return sparse_ts