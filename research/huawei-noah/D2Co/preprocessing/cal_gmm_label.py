import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import sys
sys.path.append("..")
from utils.moving_avg import freq_moving_ave, moving_ave, weighted_moving_ave

def cal_gmm_label(df_dat, windows_size, alpha):
    mm_ls = []
    min_duration = df_dat['duration_ms'].min()
    max_duration = df_dat['duration_ms'].max()
    for d in tqdm(np.arange(min_duration,max_duration+1,1)):
        X = df_dat[df_dat['duration_ms']==d]['play_time_truncate'].values
        X = X.reshape(-1,1)
        # all duration>5 
        if len(X) > 2:
            gm = GaussianMixture(n_components=2, init_params='kmeans',covariance_type='spherical', max_iter=500, random_state=61).fit(X)
            means = np.sort(gm.means_.T[0])
            stds = np.sqrt(gm.covariances_[np.argsort(gm.means_.T[0])])
            weights = gm.weights_[np.argsort(gm.means_.T[0])]
            mm_d = list(zip(means,weights))
            mm_ls.append([d, mm_d[0][0],mm_d[1][0], mm_d[1][1], stds[0], stds[1]])

    mm_ls = np.array(mm_ls)

    df_stat = df_dat[(df_dat['duration_ms']<=max_duration) & (df_dat['duration_ms']>=min_duration)]['duration_ms'].value_counts()
    freq_ls = df_stat.sort_index().values 

    nega_map_GMM_ma = dict(zip(mm_ls[:,0],freq_moving_ave(mm_ls[:,1], freq_ls, windows_size=windows_size)))
    posi_map_GMM_ma = dict(zip(mm_ls[:,0],freq_moving_ave(mm_ls[:,2], freq_ls, windows_size=windows_size)))
    weight_map = dict(zip(mm_ls[:,0],freq_moving_ave(mm_ls[:,3], freq_ls, windows_size=windows_size)))
    nega_map_std = dict(zip(mm_ls[:,0],freq_moving_ave(mm_ls[:,4], freq_ls, windows_size=windows_size)))
    posi_map_std = dict(zip(mm_ls[:,0],freq_moving_ave(mm_ls[:,5], freq_ls, windows_size=windows_size)))
    
    def cal_gmm_label(row,alpha=alpha):
        p = posi_map_GMM_ma[row['duration_ms']]
        q = nega_map_GMM_ma[row['duration_ms']]
        x = row['play_time_truncate']
        # rel = (x-q)/(p-q)
        rel = (np.exp(alpha*x) - np.exp(alpha*q)) / (np.exp(alpha*p)- np.exp(alpha*q))
        return np.clip(rel,0,1)
        # return rel

    df_dat['D2Co'] = df_dat.apply(lambda row:cal_gmm_label(row),axis=1)

    return df_dat, posi_map_GMM_ma, nega_map_GMM_ma, weight_map, nega_map_std, posi_map_std

if __name__=="__main__":
    pass