import pandas as pd
import numpy as np
from scipy.stats import norm


def cal_ground_truth(df_dat, posi_map_GMM_ma, nega_map_GMM_ma, weight_map, nega_map_std, posi_map_std, data_name):

    def decide_long(row, data_name):
        p = posi_map_GMM_ma[row['duration_ms']]
        q = nega_map_GMM_ma[row['duration_ms']]

        if data_name=='KuaiRand':
            if (row['PCR']>=1.0) & (row['duration_ms']<=18):
                return 1
            elif (row['play_time_truncate']>18) & (row['duration_ms']>18):
                return 1
            else:
                return 0
        elif data_name=='WeChat':
            if (row['PCR']>=1.0) & (row['duration_ms']<=18):
                return 1
            elif (row['play_time_truncate']>18) & (row['duration_ms']>18):
                return 1
            else:
                return 0
        
    df_dat['long_view2'] = df_dat.apply(lambda row: decide_long(row,data_name), axis=1)

    return df_dat

if __name__=="__main__":
    pass