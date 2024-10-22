import mindspore
import copy
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.metrics import NDCG
from sklearn.metrics import log_loss, mean_absolute_error


def _get_pred(data_ld, model):
    pred = []
    model.set_train(False)
    for _id, (feature , label) in enumerate(data_ld.create_tuple_iterator()):
        pred_bacth = model(feature).view(-1)
        pred_bacth = pred_bacth.asnumpy().tolist()
        pred.extend(pred_bacth)
    return pred

def cal_group_metric(df,model,posi,data_ld):
    df = copy.deepcopy(df)
    pred = _get_pred(data_ld, model)
    df['pred'] = pred
    df['rank'] = df.groupby('user_id')['pred'].rank(method='first', ascending=False)
    temp = df.groupby('user_id').apply(lambda x: x.sort_values('rank', ascending=True))
    temp.reset_index(drop=True, inplace=True)
    temp_group = temp.groupby('user_id')
    group_df = temp_group['long_view2'].apply(list).reset_index()
    group_df['pred_list'] = temp_group['pred'].apply(list).reset_index()['pred']
    # print(group_df.head(10))
    result_ls = []
    for p in posi:
        ndcg_eval = NDCG(p)
        # mrr_eval.evaluate(targets)
        ndcg_ls = group_df['long_view2'].apply(lambda x:ndcg_eval.evaluate(x))
        result_ls.append(ndcg_ls.mean())
    gauc_ls = group_df.apply(lambda row: _cal_one_usr_auc(row) ,axis=1)
    weight_ls = group_df.apply(lambda row: _cal_one_usr_weight(row) ,axis=1)
    gauc = sum(gauc_ls)/sum(weight_ls)
    return result_ls,gauc

def cal_gauc(df,model,data_ld):
    df = copy.deepcopy(df)
    pred = _get_pred(data_ld, model)
    df['pred'] = pred
    df['rank'] = df.groupby('user_id')['pred'].rank(method='first', ascending=False)
    temp = df.groupby('user_id').apply(lambda x: x.sort_values('rank', ascending=True))
    temp.reset_index(drop=True, inplace=True)
    temp_group = temp.groupby('user_id')
    group_df = temp_group['long_view2'].apply(list).reset_index()
    group_df['pred_list'] = temp_group['pred'].apply(list).reset_index()['pred']
    gauc_ls = group_df.apply(lambda row: _cal_one_usr_auc(row) ,axis=1)
    weight_ls = group_df.apply(lambda row: _cal_one_usr_weight(row) ,axis=1)
    gauc = sum(gauc_ls)/sum(weight_ls)
    return gauc

def _cal_one_usr_auc(row):
    if 0<sum(row['long_view2'])<len(row['long_view2']):
        return sum(row['long_view2']) * roc_auc_score(row['long_view2'],row['pred_list'])
    else:
        return 0

def _cal_one_usr_weight(row):
    if 0<sum(row['long_view2'])<len(row['long_view2']):
        return sum(row['long_view2'])
    else:
        return 0