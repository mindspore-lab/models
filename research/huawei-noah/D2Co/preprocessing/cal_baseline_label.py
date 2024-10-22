import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from scipy.stats import norm

def cal_baseline_label(df_dat, group_num, dat_name):

    if dat_name == 'KuaiRand':
        df_cnt_duration = df_dat[['duration_ms','play_time_truncate']].groupby('duration_ms').mean()
        df_cnt_duration.reset_index(inplace=True)
        df_cnt_duration.rename(columns={'play_time_truncate':'mean_play'},inplace = True)
        df_dat = pd.merge(df_dat, df_cnt_duration, on=['duration_ms'], how='left')

        df_cnt_duration = df_dat[['duration_ms','play_time_truncate']].groupby('duration_ms').std()
        df_cnt_duration.reset_index(inplace=True)
        df_cnt_duration.rename(columns={'play_time_truncate':'std_play'},inplace = True)
        df_dat = pd.merge(df_dat, df_cnt_duration, on=['duration_ms'], how='left')

        df_dat['gain'] = df_dat.apply(lambda row:(row['play_time_truncate'] - row['mean_play'])/row['std_play'] if row['std_play']!=0 else 0, axis=1)

    df_dat['PCR'] = df_dat.apply(lambda row: row['play_time_truncate']/row['duration_ms'], axis=1)

    df_dat['WTG'] = df_dat['gain'].apply(lambda x:norm.cdf(x))

    df_dat['quantile_bin'] = pd.qcut(df_dat['duration_ms'], group_num, labels=False, duplicates='drop') 
    temple = df_dat.groupby('quantile_bin')['play_time_truncate']
    df_ls = temple.apply(lambda x: x.to_list())
    df_dat['D2Q'] = df_dat.apply(lambda row: 0.01*percentileofscore(df_ls[row['quantile_bin']], row['play_time_truncate']),axis=1)

    df_dat['PCR_denoise'] = df_dat.apply(lambda row:0 if row['play_time_truncate']<5 else row['PCR'],axis=1)
    df_dat['WTG_denoise'] = df_dat.apply(lambda row:0 if row['play_time_truncate']<5 else row['WTG'],axis=1)
    df_dat['D2Q_denoise'] = df_dat.apply(lambda row:0 if row['play_time_truncate']<5 else row['D2Q'],axis=1)
    
    max_wt = df_dat['play_time_truncate'].max()
    min_wt = df_dat['play_time_truncate'].min()
    df_dat['scale_wt'] = df_dat['play_time_truncate'].apply(lambda x: (x-min_wt)/(max_wt - min_wt))

    return df_dat

if __name__=="__main__":
    pass