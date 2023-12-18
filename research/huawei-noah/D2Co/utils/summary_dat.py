import numpy as np
import pandas as pd

def make_feature(df):
    others = ['date','duration_ms','play_time_truncate','play_time_truncate_denoise','denoise_wt','PCR','PCR_denoise',
            'mean_play','std_play','gain','WTG', 'duration_bin',
            'mean_play_denoise','std_play_denoise','gain_denoise','WTG_denoise',
            'quantile_bin','D2Q','D2Q_denoise','D2Co','long_view2','scale_wt','wt_bin']
    fe_names = [i for i in df.columns if i not in others]
    # print(fe_names)
    return df[fe_names].values

def cal_field_dims(df):
    others = ['date','duration_ms','play_time_truncate','play_time_truncate_denoise','denoise_wt','PCR','PCR_denoise',
            'mean_play','std_play','gain','WTG', 'duration_bin',
            'mean_play_denoise','std_play_denoise','gain_denoise','WTG_denoise',
            'quantile_bin','D2Q','D2Q_denoise','D2Co','long_view2','scale_wt','wt_bin']
    fe_names = [i for i in df.columns if i not in others]
    field_dims = [len(df[fe].unique()) for fe in fe_names]
    return field_dims