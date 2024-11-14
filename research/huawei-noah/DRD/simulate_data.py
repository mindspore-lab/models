import numpy as np
import pandas as pd
import argparse
import json
import random
import os,sys
from tqdm import tqdm
from argparse import ArgumentTypeError
from sklearn.preprocessing import Normalizer 
from sklearn.preprocessing import MinMaxScaler
sys.path.append("..")
from utils.trans_format import format_trans
from utils.click_model import PositionBiasedModel
from utils.load_data import load_data_forEstimate
from utils.trans_format import format_trans
from utils.set_seed import setup_seed
import utils.click_model as CM
from simulate.simulate_click import simulateOneSession
from simulate.estimate_ips import RandomizedPropensityEstimator
from simulate.estimate_rel import RandomizedRelevanceEstimator


def simulate_data(file_path, file_path_out, session_num, isLoad, TopK, eta, noise, init_prob, rel_scale): 
    train_file = file_path + 'json_file/Train.json'
    vali_file = file_path + 'json_file/Vali.json'

    output_file_path = file_path + 'click_log/'
    if not os.path.exists(output_file_path):
        os.mkdir(output_file_path)

    # Load data from json file
    train_data = load_data_forEstimate(train_file)
    vali_data = load_data_forEstimate(vali_file)

    if isLoad:
        # Load a saved click model from json file
        with open(file_path_out + 'pbm.json') as fin:	
            clickmodel_dict = json.load(fin)
        pbm = CM.loadModelFromJson(clickmodel_dict)
        # Load a IPS Estimator from json file
        estimator = RandomizedPropensityEstimator()
        estimator.loadEstimatorFromFile(file_path_out + 'ips_estimator.json')
    else:
        # Init a click model
        pbm = CM.PositionBiasedModel(eta=eta, TopK=TopK, pos_click_prob=1.0, neg_click_prob=noise, initial_incorrect_click_p=init_prob, relevance_grading_num=rel_scale)
        pbm.setExamProb(eta)
        pbm.setClickProb(noise, 1.0, rel_scale)
        pbm.setTrustProb(initial_incorrect_click_p=init_prob)
        pbm.outputModelJson(file_path_out + 'pbm.json')
        # Init a IPS Estimator 
        estimator = RandomizedPropensityEstimator()
        print('Estimating IPS from radomized click session...')
        estimator.estimateParametersFromModel(pbm, train_data, 10e5)
        estimator.outputEstimatorToFile(file_path_out + 'ips_estimator.json')

    print('Simulate click sessions...')

    # simulate on train data
    overallSessionLog = []
    for sid in tqdm(range(int(session_num))):
        #queryID = random.randint(0,train_data.rank_list_size - 1)
        queryID = np.random.choice(train_data.qid_lists, 1, replace=True)[0]
        query_list = train_data.query_lists[queryID]
        oneSessionLog = simulateOneSession(pbm, query_list)
        ips_list = estimator.getPropensityForOneList([d['isClick'] for d in oneSessionLog])
        ips_for_train_list = estimator.getPropensityForOneList_ForTrain([d['isClick'] for d in oneSessionLog])
        for index, doc in enumerate(oneSessionLog):
            doc['ips_weight'] = ips_list[index]
            doc['ips_weight_for_train'] = ips_for_train_list[index]
            doc['sid'] = sid
        overallSessionLog.extend(oneSessionLog)

    with open(output_file_path + 'Train_log.json', 'w') as fout:
        fout.write(json.dumps(overallSessionLog))

    # simulate on vali data
    # overallSessionLog = []
    # vali_session_num = session_num * (vali_data.data_size / train_data.data_size)
    # for sid in tqdm(range(int(vali_session_num))):
    #     #queryID = random.randint(0,vali_data.rank_list_size - 1)
    #     queryID = np.random.choice(vali_data.qid_lists, 1, replace=True)[0]
    #     query_list = vali_data.query_lists[queryID]
    #     oneSessionLog = simulateOneSession(pbm, query_list)
    #     ips_list = estimator.getPropensityForOneList([d['isClick'] for d in oneSessionLog])
    #     ips_for_train_list = estimator.getPropensityForOneList_ForTrain([d['isClick'] for d in oneSessionLog])
    #     for index, doc in enumerate(oneSessionLog):
    #         doc['ips_weight'] = ips_list[index]
    #         doc['ips_weight_for_train'] = ips_for_train_list[index]
    #         doc['sid'] = sid
    #     overallSessionLog.extend(oneSessionLog)

    # with open(output_file_path + 'Vali_log.json', 'w') as fout:
    #     fout.write(json.dumps(overallSessionLog))


def merge_dat(fin):
    train_log = pd.read_json(fin + 'click_log/Train_log.json')
    train_dat = pd.read_json(fin + 'json_file/Train.json')
    train_dat = format_trans(train_dat)
    click_p = train_log[['did','isClick']].groupby('did').mean()
    click_p.rename(columns={'isClick':'Click'},inplace = True)
    train_log_fe = pd.merge(train_dat,click_p,how='left',on=['did'])
    if 'WEB' in fin:
        print('Normalizing feature...')
        test_dat = pd.read_json(fin + 'json_file/Test.json')
        vali_dat = pd.read_json(fin + 'json_file/Vali.json')
        # test_dat = format_trans(test_dat)
        train_log_fe, test_dat, vali_dat = norm_feature(train_log_fe, test_dat, vali_dat)
        test_dat.to_json(fin + 'json_file/Test.json')
        vali_dat.to_json(fin + 'json_file/Vali.json')
    train_log_fe.to_json(fin + 'click_log/Train_log_trans.json')
    
    return 0

def norm_feature(df, df_test, df_vali):
    arr = np.array(df['feature'].tolist())
    arr_test = np.array(df_test['feature'].tolist())
    arr_vali = np.array(df_vali['feature'].tolist())

    scaler = MinMaxScaler()
    scaler.fit(arr)
    # arr_norm = Normalizer(norm='l2').fit_transform(arr)
    arr_norm = scaler.transform(arr)
    arr_test_norm = scaler.transform(arr_test)
    arr_vali_norm = scaler.transform(arr_vali)

    df['norm_feature'] = arr_norm.tolist()
    df.drop('feature',axis=1, inplace=True)
    df.rename(columns={'norm_feature':'feature'}, inplace = True)

    df_test['norm_feature'] = arr_test_norm.tolist()
    df_test.drop('feature',axis=1, inplace=True)
    df_test.rename(columns={'norm_feature':'feature'}, inplace = True)

    df_vali['norm_feature'] = arr_vali_norm.tolist()
    df_vali.drop('feature',axis=1, inplace=True)
    df_vali.rename(columns={'norm_feature':'feature'}, inplace = True)

    return df, df_test, df_vali


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', required=True)
    parser.add_argument('--fp2', required=True)
    parser.add_argument('--session_num', type=float, default=1e3)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--init_prob', type=float, default=0.638)
    parser.add_argument('--rel_scale', type=int, default=1)
    parser.add_argument('--TopK', type=int, default=10)
    parser.add_argument('--isLoad',type=str2bool, nargs='?', default=False)

    args = parser.parse_args()
    setup_seed(41)
    simulate_data(args.fp, args.fp2, session_num=args.session_num, isLoad=args.isLoad, TopK=args.TopK, eta=args.eta, noise=args.noise, init_prob=args.init_prob, rel_scale=args.rel_scale)
    merge_dat(args.fp)


