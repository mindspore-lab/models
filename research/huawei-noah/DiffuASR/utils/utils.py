# -*- encoding: utf-8 -*-
# here put the import lib
import os
import random
import numpy as np
import pandas as pd
import mindspore
from tqdm import tqdm


def set_seed(seed):
    '''Fix all of random seed for reproducible training'''
    random.seed(seed)
    np.random.seed(seed)
    #mindspore.manual_seed(seed)
    # mindspore.cuda.manual_seed(seed)
    # mindspore.cuda.manual_seed_all(seed)
    # mindspore.backends.cudnn.deterministic = True   # only add when conv in your model
    mindspore.set_seed(seed)


def get_n_params(model):
    '''Get the number of parameters of model'''
    pp = 0
    for p in list(model.get_parameters()):
        nn = 1
        for s in list(p.shape):
            nn = nn*s
        pp += nn
    return pp


def get_n_params_(parameter_list):
    '''Get the number of parameters of model'''
    pp = 0
    for p in list(parameter_list):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def unzip_data(data, aug=True, aug_num=0):

    res = []
    
    if aug:
        for user in tqdm(data):

            user_seq = data[user]
            seq_len = len(user_seq)

            for i in range(aug_num+2, seq_len+1):
                
                res.append(user_seq[:i])
    else:
        for user in tqdm(data):

            user_seq = data[user]
            res.append(user_seq)

    return res


def unzip_data_with_user(data, aug=True, aug_num=0):

    res = []
    users = []
    user_id = 1
    
    if aug:
        for user in tqdm(data):

            user_seq = data[user]
            seq_len = len(user_seq)

            for i in range(aug_num+2, seq_len+1):
                
                res.append(user_seq[:i])
                users.append(user_id)

            user_id += 1

    else:
        for user in tqdm(data):

            user_seq = data[user]
            res.append(user_seq)
            users.append(user_id)
            user_id += 1

    return res, users


def concat_data(data_list):

    res = []

    if len(data_list) == 2:

        train = data_list[0]
        valid = data_list[1]

        for user in train:

            # if len(valid[user]) == 0:
                
            #     continue
            
            # else:
                
            #     res.append(train[user]+valid[user])
            res.append(train[user]+valid[user])
    
    elif len(data_list) == 3:

        train = data_list[0]
        valid = data_list[1]
        test = data_list[2]

        for user in train:

            res.append(train[user]+valid[user]+test[user])

    else:

        raise ValueError

    return res


def concat_aug_data(data_list):

    res = []

    train = data_list[0]
    valid = data_list[1]

    for user in train:

        if len(valid[user]) == 0:
            res.append([train[user][0]])
        
        else:
            res.append(train[user]+valid[user])

    return res


def concat_data_with_user(data_list):

    res = []
    users = []
    user_id = 1

    if len(data_list) == 2:

        train = data_list[0]
        valid = data_list[1]

        for user in train:

            res.append(train[user]+valid[user])
            users.append(user_id)
            user_id += 1
    
    elif len(data_list) == 3:

        train = data_list[0]
        valid = data_list[1]
        test = data_list[2]

        for user in train:

            res.append(train[user]+valid[user]+test[user])
            users.append(user_id)
            user_id += 1

    else:

        raise ValueError

    return res, users


def filter_data(data, thershold=5):
    '''Filter out the sequence shorter than threshold'''
    res = []

    for user in data:

        if len(user) > thershold:
            res.append(user)
        else:
            continue
    
    return res



def random_neq(l, r, s=[]):    # 在l-r之间随机采样一个数，这个数不能在列表s中
    
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t



def metric_report(data_rank, topk=10):

    NDCG, HT = 0, 0
    
    for rank in data_rank:

        if rank < topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return {'NDCG@10': NDCG / len(data_rank),
            'HR@10': HT / len(data_rank)}



def metric_len_report(data_rank, data_len, topk=10, aug_len=0):

    NDCG_s, HT_s = 0, 0
    NDCG_m, HT_m = 0, 0
    NDCG_l, HT_l = 0, 0
    count_s = len(data_len[data_len<=5+aug_len])
    count_l = len(data_len[data_len>20+aug_len])
    count_m = len(data_len) - count_s - count_l

    for i, rank in enumerate(data_rank):

        if rank < topk:

            if data_len[i] <= 5+aug_len:
                NDCG_s += 1 / np.log2(rank + 2)
                HT_s += 1
            elif data_len[i] <= 20+aug_len:
                NDCG_m += 1 / np.log2(rank + 2)
                HT_m += 1
            else:
                NDCG_l += 1 / np.log2(rank + 2)
                HT_l += 1

    return {'Short NDCG@10': NDCG_s / count_s,
            'Short HR@10': HT_s / count_s,
            'Medium NDCG@10': NDCG_m / count_m,
            'Medium HR@10': HT_m / count_m,
            'Long NDCG@10': NDCG_l / count_l,
            'Long HR@10': HT_l / count_l,}


def seq_acc(true, pred):

    true_num = np.sum((true==pred))
    total_num = true.shape[0] * true.shape[1]

    return {'acc': true_num / total_num}


def load_pretrained_model(pretrain_dir, model, logger):

    logger.info("Loading pretrained model ... ")
    checkpoint_path = os.path.join(pretrain_dir, 'pytorch_model.bin.ckpt')

    model_dict = model.parameters_dict()

    # To be compatible with the new and old version of model saver
    pretrained_dict = mindspore.load_checkpoint(checkpoint_path)

    # filter out required parameters
    new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    # 打印出来，更新了多少的参数
    logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
    mindspore.load_param_into_net(model, model_dict)

    return model


def record_csv(args, res_dict, path='log'):
    
    path = os.path.join(path, args.dataset)

    if not os.path.exists(path):
        os.makedirs(path)

    record_file = args.model_name + '.csv'
    csv_path = os.path.join(path, record_file)
    model_name = args.aug_file + '-' + args.now_str
    res_dict["model_name"] = model_name
    columns = ["model_name", "HR@10", "NDCG@10", "Short HR@10", "Short NDCG@10", "Medium HR@10", "Medium NDCG@10", "Long HR@10", "Long NDCG@10",]
    new_res_dict = {key: [value] for key, value in res_dict.items()}
    
    if not os.path.exists(csv_path):

        df = pd.DataFrame(new_res_dict)
        df = df[columns]    # reindex the columns
        df.to_csv(csv_path, index=False)

    else:

        df = pd.read_csv(csv_path)
        add_df = pd.DataFrame(new_res_dict)
        df = pd.concat([df, add_df])
        df.to_csv(csv_path, index=False)
