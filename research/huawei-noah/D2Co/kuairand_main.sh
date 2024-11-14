#!/bin/bash
set -e
set -x

randseed=61
dataname=KuaiRand
windows_size=3
alpha=-0.05


python prepare_data.py --group_num 120 --windows_size ${windows_size} --alpha ${alpha} --dat_name ${dataname} --is_load 0

for labelname in long_view2 scale_wt PCR PCR_denoise D2Q D2Q_denoise WTG WTG_denoise D2Co 
do 
    python main.py --fout ../rec_datasets/Duration_Test/${modelname}_${labelname}_${windows_size}_${alpha} --dat_name ${dataname} --model_name ${modelname} --label_name ${labelname} --windows_size ${windows_size} --alpha ${alpha}
done 

