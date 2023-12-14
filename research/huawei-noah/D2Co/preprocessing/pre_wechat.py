import pandas as pd 
import numpy as np

def pre_wechat():
    df_wechat = pd.read_csv('../rec_datasets/wechat_video_debias.csv')
    df_wechat.rename(columns={'playseconds_truncate':'play_time_truncate',
                          'videoplayseconds':'duration_ms',
                          'date_':'date'},inplace=True)

    df_wechat['bgm_song_id'] =  df_wechat['bgm_song_id'].apply(lambda x: 99999 if pd.isna(x) else x)
    df_wechat['bgm_singer_id'] =  df_wechat['bgm_song_id'].apply(lambda x: 99999 if pd.isna(x) else x)

    userid_map = dict(zip(np.sort(df_wechat['userid'].unique()),range(len(df_wechat['userid'].unique()))))
    feedid_map = dict(zip(np.sort(df_wechat['feedid'].unique()),range(len(df_wechat['feedid'].unique()))))
    authorid_map = dict(zip(np.sort(df_wechat['authorid'].unique()),range(len(df_wechat['authorid'].unique()))))
    bgm_song_id_map = dict(zip(np.sort(df_wechat['bgm_song_id'].unique()),range(len(df_wechat['bgm_song_id'].unique()))))
    bgm_singer_id_map = dict(zip(np.sort(df_wechat['bgm_singer_id'].unique()),range(len(df_wechat['bgm_singer_id'].unique()))))


    df_wechat['userid'] = df_wechat['userid'].apply(lambda x: userid_map[x])
    df_wechat['feedid'] = df_wechat['feedid'].apply(lambda x: feedid_map[x])
    df_wechat['authorid'] = df_wechat['authorid'].apply(lambda x: authorid_map[x])
    df_wechat['bgm_song_id'] = df_wechat['bgm_song_id'].apply(lambda x: bgm_song_id_map[x])
    df_wechat['bgm_singer_id'] = df_wechat['bgm_singer_id'].apply(lambda x: bgm_singer_id_map[x])

    df_sel_dat = df_wechat[['date','userid', 'feedid', 'device', 'authorid',
                            'bgm_song_id', 'bgm_singer_id','user_type', 'feed_type', 'like', 'read_comment',
                            'forward','duration_ms','play_time_truncate', 'mean_play', 'std_play', 'gain']]

    df_sel_dat.rename(columns={'userid':'user_id','feedid':'video_id'}, inplace=True)
    
    return df_sel_dat

