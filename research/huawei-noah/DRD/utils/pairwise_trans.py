import numpy as np
import pandas as pd
from tqdm import tqdm

def ipw(row):
    return row['isClick'] * row['ips_weight']

def sn_ipw(row, sn_factor):
    return row['isClick'] * (row['ips_weight'] / sn_factor)

def log_smooth_ipw(row):
    return np.log(1 + row['isClick'] * row['ips_weight'])

def get_pair_sel(df, mode='train', topk=5):
    if mode == 'train':
        df_pair = pd.DataFrame([],columns=['sid', 'qid', 'rel_diff', 'click_diff',
                                'pos_did', 'neg_did','pos_pos','neg_pos','pos_sel','neg_sel','tag'])
    elif mode == 'vali':
        # df_pair = pd.DataFrame([],columns=['sid', 'qid', 'rel_diff', 'click_diff',
        #                         'pos_did', 'pos_feature', 'neg_did', 'neg_feature','pos_pos','neg_pos','pos_sel','neg_sel','tag'])
        df_pair = pd.DataFrame([],columns=['sid', 'qid', 'rel_diff', 'click_diff',
                                'pos_did', 'neg_did','pos_pos','neg_pos','pos_sel','neg_sel','tag'])
    df_top_k = df[df['rankPosition']<topk]
    df_out_k = df[df['rankPosition']>=topk]
    df_unique_out_k = df_out_k.drop_duplicates('did')
    df_new = pd.concat([df_top_k, df_unique_out_k])
    # df_topk = df[df['rankPosition']<topk]
    # df_outk = df[df['rankPosition']>=topk]
    df_sort = df_new.groupby('sid').apply(lambda x: x.sort_values(by = ['isClick','ips_weight'], ascending=[False, False]))
    df_sort.reset_index(drop=True, inplace=True)   
    # sn_factor = df_sort[df_sort['isClick']==1]['ips_weight'].mean() 
    # print('sn_factor:',sn_factor)
    # sn_factor_true = len(df_sort[df_sort['isClick']==1])/np.sum(np.power(df_sort[df_sort['isClick']==1]['ips_weight'].values,-1))
    # print('sn_factor_true:',sn_factor_true)
    rows_pair = []

    for _, row_out in tqdm(df_sort.iterrows(), total=df_sort.shape[0]):
        ipw_row_out = ipw(row_out)
        for _, row_in in df_sort[df_sort['sid']==row_out['sid']].iterrows():
            ipw_row_in = ipw(row_in)
            if (row_out['rankPosition']<topk) and (row_in['rankPosition']<topk) and (ipw_row_out>ipw_row_in):
                if mode == 'train':
                        row_new = [row_out['sid'], row_out['qid'], 
                                        ipw_row_out - ipw(row_in),
                                        # 1, 
                                        row_out['isClick'] - row_in['isClick'],
                                        # row_out['isClick'],
                                        row_out['did'], 
                                        row_in['did'],
                                        row_out['rankPosition'],
                                        row_in['rankPosition'],
                                        row_out['isSelect'],
                                        row_in['isSelect'],
                                        1]
                elif mode == 'vali':
                    row_new = [row_out['sid'], row_out['qid'], 
                                        ipw_row_out - ipw(row_in),
                                        # 1, 
                                        row_out['isClick'] - row_in['isClick'],
                                        # row_out['isClick'],
                                        row_out['did'], 
                                        # row_out['feature'], 
                                        row_in['did'], 
                                        # row_in['feature'],
                                        row_out['rankPosition'],
                                        row_in['rankPosition'],
                                        row_out['isSelect'],
                                        row_in['isSelect'],
                                        1]
                rows_pair.append(row_new)
            else:
                # if np.random.rand()<0.1:
                #     pass
                # else:
                #     pass
                if mode == 'train':
                    row_new = [row_out['sid'], row_out['qid'], 
                                        0,
                                        0,
                                        row_out['did'], 
                                        row_in['did'],
                                        row_out['rankPosition'],
                                        row_in['rankPosition'],
                                        row_out['isSelect'],
                                        row_in['isSelect'],
                                        0]
                elif mode == 'vali':
                    row_new = [row_out['sid'], row_out['qid'], 
                                        0,
                                        0,
                                        row_out['did'], 
                                        # row_out['feature'], 
                                        row_in['did'], 
                                        # row_in['feature'],
                                        row_out['rankPosition'],
                                        row_in['rankPosition'],
                                        row_out['isSelect'],
                                        row_in['isSelect'],
                                        0]
                rows_pair.append(row_new)

    rows_pair = np.array(rows_pair, dtype=object)
    for i, col_name in enumerate(df_pair.columns):
        df_pair[col_name] = rows_pair[:,i]
    return df_pair


def get_pair(df, mode='train'):
    if mode == 'train':
        df_pair = pd.DataFrame([],columns=['sid', 'qid', 'rel_diff', 'click_diff',
                                'pos_did', 'neg_did','pos_pos','neg_pos','pos_sel','neg_sel'])
    elif mode == 'vali':
        # df_pair = pd.DataFrame([],columns=['sid', 'qid', 'rel_diff', 'click_diff',
        #                         'pos_did', 'pos_feature', 'neg_did', 'neg_feature','pos_pos','neg_pos','pos_sel','neg_sel'])
        df_pair = pd.DataFrame([],columns=['sid', 'qid', 'rel_diff', 'click_diff',
                                'pos_did', 'neg_did','pos_pos','neg_pos','pos_sel','neg_sel'])

    df_sort = df.groupby('sid').apply(lambda x: x.sort_values(by = ['isClick','ips_weight'], ascending=[False, False]))
    df_sort.reset_index(drop=True, inplace=True)   
    # sn_factor = df_sort[df_sort['isClick']==1]['ips_weight'].mean() 
    # print('sn_factor:',sn_factor)
    # sn_factor_true = len(df_sort[df_sort['isClick']==1])/np.sum(np.power(df_sort[df_sort['isClick']==1]['ips_weight'].values,-1))
    # print('sn_factor_true:',sn_factor_true)
    rows_pair = []
    for _, row_out in tqdm(df_sort.iterrows(), total=df_sort.shape[0]):
        if row_out['isClick'] > 0:
            ipw_row_out = ipw(row_out)
            for _, row_in in df_sort[df_sort['sid']==row_out['sid']].iterrows():
                if ipw_row_out > ipw(row_in):
                    if mode == 'train':
                        row_new = [row_out['sid'], row_out['qid'], 
                                        ipw_row_out - ipw(row_in),
                                        # 1, 
                                        row_out['isClick'] - row_in['isClick'],
                                        # row_out['isClick'],
                                        row_out['did'], 
                                        row_in['did'],
                                        row_out['rankPosition'],
                                        row_in['rankPosition'],
                                        row_out['isSelect'],
                                        row_in['isSelect']]
                    elif mode == 'vali':
                        row_new = [row_out['sid'], row_out['qid'], 
                                            ipw_row_out - ipw(row_in),
                                            # 1, 
                                            row_out['isClick'] - row_in['isClick'],
                                            # row_out['isClick'],
                                            row_out['did'],
                                            # row_out['feature'], 
                                            row_in['did'],
                                            # row_in['feature'],
                                            row_out['rankPosition'],
                                            row_in['rankPosition'],
                                            row_out['isSelect'],
                                            row_in['isSelect']]
                    rows_pair.append(row_new)
        else:
            continue


    # for _, row_out in tqdm(df_sort.iterrows(), total=df_sort.shape[0]):
    #     if row_out['isClick'] > 0:
    #         ipw_row_out = ipw(row_out) 
    #         for _, row_in in df_sort[df_sort['sid']==row_out['sid']].iterrows():
    #             if mode == 'train':
    #                 row_new = [row_out['sid'], row_out['qid'], 
    #                                 ipw_row_out, 
    #                                 row_out['isClick'] - row_in['isClick'],
    #                                 row_out['did'], 
    #                                 row_in['did'],
    #                                 row_out['rankPosition'],
    #                                 row_in['rankPosition']]
    #             elif mode == 'vali':
    #                 row_new = [row_out['sid'], row_out['qid'], 
    #                                     ipw_row_out, 
    #                                     row_out['isClick'] - row_in['isClick'],
    #                                     row_out['did'], row_out['feature'], 
    #                                     row_in['did'], row_in['feature'],
    #                                     row_out['rankPosition'],
    #                                     row_in['rankPosition']]
    #             rows_pair.append(row_new)
    #     else:
    #         continue
        

    rows_pair = np.array(rows_pair, dtype=object)
    for i, col_name in enumerate(df_pair.columns):
        df_pair[col_name] = rows_pair[:,i]
    return df_pair


def get_pair_fullinfo(df, mode='train'):
    if mode=='train':
        df_pair = pd.DataFrame([],columns=['qid', 'rel_diff', 'pos_did', 'neg_did'])
    elif mode=='vali':
        # df_pair = pd.DataFrame([],columns=['qid', 'rel_diff', 'pos_did', 'pos_feature', 'neg_did', 'neg_feature'])
        df_pair = pd.DataFrame([],columns=['qid', 'rel_diff', 'pos_did', 'neg_did'])
    df_sort = df.groupby('qid').apply(lambda x: x.sort_values(by = ['label'], ascending=[False]))
    df_sort.reset_index(drop=True, inplace=True)    
    rows_pair = []
    for _, row_out in tqdm(df_sort.iterrows(), total=df_sort.shape[0]):
        if row_out['label'] > 0: 
            for _, row_in in df_sort[df_sort['qid']==row_out['qid']].iterrows():
                if row_out['label'] > row_in['label']:
                    if mode == 'train':
                        row_new = [ row_out['qid'], row_out['label'] - row_in['label'], 
                                            row_out['did'], 
                                            row_in['did']]
                    elif mode == 'vali':
                        row_new = [ row_out['qid'], row_out['label'] - row_in['label'], 
                                            row_out['did'], 
                                            # row_out['feature'], 
                                            row_in['did'], 
                                            # row_in['feature']
                                            ]
                    rows_pair.append(row_new)
        else:
            continue

    rows_pair = np.array(rows_pair, dtype=object)
    for i, col_name in enumerate(df_pair.columns):
        df_pair[col_name] = rows_pair[:,i]
    return df_pair


if __name__=="__main__":
    train_log = pd.read_json('./datasets/MQ2008/' + 'click_log/Train_log.json')
    train_pair_log = get_pair(train_log)
    print(train_pair_log.head(1))