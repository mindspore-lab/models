import argparse
import numpy as np
import pandas as pd
from argparse import ArgumentTypeError
from preprocessing.pre_kuairand import pre_kuairand
from preprocessing.pre_wechat import pre_wechat
from preprocessing.cal_baseline_label import cal_baseline_label
from preprocessing.cal_gmm_label import cal_gmm_label
from preprocessing.cal_ground_truth import cal_ground_truth

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="prepare datasets")
    #parser.add_argument('-f', '--file_path', dest='file_path', required=True)
    parser.add_argument('-g', '--group_num', type=int, default=30, help="Groups of D2Q")
    parser.add_argument('-t', '--windows_size', type=int, default=10, help='Windows size of moving average')
    parser.add_argument('-e', '--alpha', type=float, default=0.3, help='sensitivity control term')
    parser.add_argument('--dat_name', type=str, default='KuaiRand', choices=['KuaiRand', 'WeChat'])
    parser.add_argument('--is_load', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()

    #file_path = args.file_path
    group_num = args.group_num
    dat_name = args.dat_name
    windows_size = args.windows_size
    alpha = args.alpha
    is_load = args.is_load

    if dat_name=='KuaiRand':
        if is_load == True:
            print('Load Processed Data...')
            kuairand_dat = pd.read_json('../rec_datasets/Duration_KuaiRand/KuaiRand_subset.json')
            print('Cal GMM Labels...')
            kuairand_dat, posi_map_GMM_ma, nega_map_GMM_ma, weight_map, nega_map_std, posi_map_std = cal_gmm_label(kuairand_dat, windows_size, alpha)


            kuairand_dat.to_json('../rec_datasets/Duration_KuaiRand/KuaiRand_subset.json')
            print(kuairand_dat.head(10))
        else:
            print('Load Raw Data...')
            kuairand_dat = pre_kuairand()
            
            print('Cal Baseline Labels...')
            kuairand_dat = cal_baseline_label(kuairand_dat, group_num, dat_name)
            
            print('Cal GMM Labels...')
            kuairand_dat, posi_map_GMM_ma, nega_map_GMM_ma, weight_map, nega_map_std, posi_map_std = cal_gmm_label(kuairand_dat, windows_size, alpha)

            print('Cal Ground Truth Labels...')
            kuairand_dat = cal_ground_truth(kuairand_dat, posi_map_GMM_ma, nega_map_GMM_ma, weight_map, nega_map_std, posi_map_std, dat_name)
            kuairand_dat.to_json('../rec_datasets/Duration_KuaiRand/KuaiRand_subset.json')
            print(kuairand_dat.head(10))


    elif dat_name == 'WeChat':
        if is_load == True:
            print('Load Processed Data...')
            wechat_dat = pd.read_json('../rec_datasets/Duration_WeChat/WeChat_subset.json')
            print(len(wechat_dat))
            print('Cal GMM Labels...')
            wechat_dat, posi_map_GMM_ma, nega_map_GMM_ma, weight_map, nega_map_std, posi_map_std = cal_gmm_label(wechat_dat, windows_size, alpha)

            wechat_dat.to_json('../rec_datasets/Duration_WeChat/WeChat_subset.json')
            print(wechat_dat.head(10))
        else:
            print('Load Raw Data...')
            wechat_dat = pre_wechat()
            print('Cal Baseline Labels...')
            wechat_dat = cal_baseline_label(wechat_dat, group_num, dat_name)
            print('Cal GMM Labels...')
            wechat_dat, posi_map_GMM_ma, nega_map_GMM_ma,weight_map, nega_map_std, posi_map_std = cal_gmm_label(wechat_dat, windows_size, alpha)
            print('Cal Ground Truth Labels...')
            wechat_dat = cal_ground_truth(wechat_dat, posi_map_GMM_ma, nega_map_GMM_ma, weight_map, nega_map_std, posi_map_std, dat_name)
            wechat_dat.to_json('../rec_datasets/Duration_WeChat/WeChat_subset.json')
            print(wechat_dat.head(10))


if __name__ == "__main__":
    main()