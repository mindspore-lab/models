import pandas as pd 
import numpy as np

def contain_ls(ls):
    result_ls = []
    for x in ls:
        result_ls.extend(x)
    return result_ls

def compare_max(cat_ls, frac_dict):
    frac_ls = np.array([frac_dict[c] for c in cat_ls])
    cat_ls = np.array(cat_ls)
    frac_sort_cat_ls = cat_ls[np.argsort(frac_ls)][::-1]
    return frac_sort_cat_ls[0]


def pre_kuairand():
    #KuaiRand-Pure
    df_kuaiRand_interaction_1 = pd.read_csv('../rec_datasets/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv')
    df_kuaiRand_interaction_2 = pd.read_csv('../rec_datasets/KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv')
    df_kuaiRand_video_fe_basic = pd.read_csv('../rec_datasets/KuaiRand-Pure/data/video_features_basic_pure.csv')

    df_kuaiRand_interaction_standard = df_kuaiRand_interaction_1.append(df_kuaiRand_interaction_2)

    df_kuaiRand_interaction_standard['play_time_truncate'] = df_kuaiRand_interaction_standard.apply(lambda row:row['play_time_ms'] if row['play_time_ms']<row['duration_ms'] else row['duration_ms'],axis=1)
    df_kuaiRand_interaction_standard['duration_ms'] = df_kuaiRand_interaction_standard['duration_ms'].apply(lambda x: np.round(x/1e3))
    df_kuaiRand_interaction_standard['play_time_truncate'] = df_kuaiRand_interaction_standard['play_time_truncate'].apply(lambda x: np.round(x/1e3))

    dic_video_type = {'NORMAL':1,'AD':0,'UNKNOWN':0}
    df_kuaiRand_video_fe_basic['video_type'] = df_kuaiRand_video_fe_basic['video_type'].apply(lambda x: dic_video_type[x])

    dic_upload_type = {'LongImport':0,
                    'ShortImport':1,
                    'Web':2,
                    'Kmovie':3,
                    'LongPicture':4,
                    'PictureSet':5,
                    'LongCamera':6,
                    'ShortCamera':7,
                    'ShareFromOtherApp':8,
                    'FollowShoot':9,
                    'AiCutVideo':10,
                    'LipsSync':11,
                    'PhotoCopy':12,
                    'UNKNOWN':-1,}
    df_kuaiRand_video_fe_basic['upload_type'] = df_kuaiRand_video_fe_basic['upload_type'].apply(lambda x: dic_upload_type[x])

    df_kuaiRand_video_fe_basic['tag_ls'] = df_kuaiRand_video_fe_basic['tag'].apply(lambda x: str(x).split(','))

    total_ls = contain_ls(df_kuaiRand_video_fe_basic['tag_ls'].values)
    stat_series = pd.Series(total_ls).value_counts()
    count_info = dict(zip(stat_series.index,stat_series.values))
    df_kuaiRand_video_fe_basic['tag_pop'] = df_kuaiRand_video_fe_basic['tag_ls'].apply(lambda x: compare_max(x, count_info))

    df_kuaiRand_interaction_standard = pd.merge(df_kuaiRand_interaction_standard, df_kuaiRand_video_fe_basic, on=['video_id'], how='left')

    # select duration range and featrues
    df_sel_dat = df_kuaiRand_interaction_standard[(df_kuaiRand_interaction_standard['duration_ms']<=240) & (df_kuaiRand_interaction_standard['duration_ms']>=5)]
    df_sel_dat = df_sel_dat[['date','user_id','video_id','author_id','music_id','tag_pop','video_type','upload_type',
                            'tab','is_like','is_follow','is_comment','is_forward','is_profile_enter','is_hate','duration_ms','play_time_truncate']]
    df_sel_dat['tag_pop'] =  df_sel_dat['tag_pop'].apply(lambda x: 999 if pd.isna(x) else x)

    user_id_map = dict(zip(np.sort(df_sel_dat['user_id'].unique()),range(len(df_sel_dat['user_id'].unique()))))
    video_id_map = dict(zip(np.sort(df_sel_dat['video_id'].unique()),range(len(df_sel_dat['video_id'].unique()))))
    author_id_map = dict(zip(np.sort(df_sel_dat['author_id'].unique()),range(len(df_sel_dat['author_id'].unique()))))
    music_id_map = dict(zip(np.sort(df_sel_dat['music_id'].unique()),range(len(df_sel_dat['music_id'].unique()))))
    tag_pop_map = dict(zip(np.sort(df_sel_dat['tag_pop'].unique()),range(len(df_sel_dat['tag_pop'].unique()))))
    upload_type_map = dict(zip(np.sort(df_sel_dat['upload_type'].unique()),range(len(df_sel_dat['upload_type'].unique()))))

    df_sel_dat['user_id'] = df_sel_dat['user_id'].apply(lambda x: user_id_map[x])
    df_sel_dat['video_id'] = df_sel_dat['video_id'].apply(lambda x: video_id_map[x])
    df_sel_dat['author_id'] = df_sel_dat['author_id'].apply(lambda x: author_id_map[x])
    df_sel_dat['music_id'] = df_sel_dat['music_id'].apply(lambda x: music_id_map[x])
    df_sel_dat['tag_pop'] = df_sel_dat['tag_pop'].apply(lambda x: tag_pop_map[x])
    df_sel_dat['upload_type'] = df_sel_dat['upload_type'].apply(lambda x: upload_type_map[x])

    return df_sel_dat

if __name__=="__main__":
    pass