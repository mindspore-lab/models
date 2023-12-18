import numpy as np
import pandas as pd


def weighted_moving_ave(ls, windows_size=5):
    new_ls = []
    for idx,val in enumerate(ls):
        if idx < windows_size - 1:
            weight_term_ls = list(range(windows_size-idx,windows_size)) + list(range(1,windows_size+1))[::-1]
            ave_term_ls = ls[0: idx + windows_size]
            weight_sum = sum(np.array(weight_term_ls) * np.array(ave_term_ls))
            #normalize_term = 0.5 * windows_size * (windows_size + 1)
            normalize_term = sum(weight_term_ls)
            weight_ave = weight_sum/normalize_term

        elif idx > len(ls) - windows_size:
            weight_term_ls = list(range(1,windows_size)) + list(range(windows_size+1-len(ls)+idx,windows_size+1))[::-1]
            ave_term_ls = ls[idx - windows_size + 1: len(ls)]
            weight_sum = sum(np.array(weight_term_ls) * np.array(ave_term_ls))
            #normalize_term = 0.5 * windows_size * (windows_size + 1)
            normalize_term = sum(weight_term_ls)
            weight_ave = weight_sum/normalize_term
            
        else:
            weight_term_ls = list(range(1,windows_size)) + list(range(1,windows_size+1))[::-1]
            ave_term_ls = ls[idx - windows_size + 1: idx + windows_size] # ws>1
            weight_sum = sum(np.array(weight_term_ls) * np.array(ave_term_ls))
            #normalize_term = 0.5 * windows_size * (windows_size + 1)
            normalize_term = sum(weight_term_ls)
            weight_ave = weight_sum/normalize_term

        new_ls.append(weight_ave)
    return new_ls

def moving_ave(ls, windows_size=5):
    amount = pd.Series(ls)
    ave_result = amount.rolling(2*windows_size-1, min_periods=1, center=True).agg(lambda x: np.mean(x))
    return ave_result.values

def freq_moving_ave(ls_v, ls_w, windows_size=5):
    ls_mul = np.array(ls_v) * np.array(ls_w)
    amount = pd.Series(ls_mul)
    amount_sum = amount.rolling(2*windows_size-1, min_periods=1, center=True).agg(lambda x: np.sum(x))
    
    weight = pd.Series(ls_w)
    weight_sum = weight.rolling(2*windows_size-1, min_periods=1, center=True).agg(lambda x: np.sum(x))
    
    return amount_sum/weight_sum

if __name__=="__main__":
    pass
