#!/bin/bash
set -e
set -x

TopK=10
session_num=1e6
eta=1.0 
init_prob=0.638
randseed=53
dataname="YaHooC14B_main"
output_folder="YaHooC14B_drd_main"

echo "dataset: ${dataname}"
echo "output_folder: ${output_folder}"
echo "session_num: ${session_num}"
echo "TopK: ${TopK}"
echo "eta: ${eta}"
echo "init_prob: ${init_prob}"

# python prepare_data.py --file_path ../datasets/${dataname}/  --proportion 0.01

# python simulate_data.py  --fp ../datasets/${dataname}/ --fp2 ../datasets/${output_folder}/ --session_num ${session_num} --eta ${eta} --isLoad 0 --TopK ${TopK} --rel_scale 4 --init_prob ${init_prob}

# CUDA_VISIBLE_DEVICES=0 python main.py --fin ../datasets/${dataname}/ --fout ../datasets/${output_folder}/drd --train_alg drd --pairwise 1 --lr 1e-3 --weight_decay 0 --batch_size 128  --topK ${TopK} --epoch 50  --randseed ${randseed} --drop_out 0.1 --eta ${eta}  --alpha 1.2  --min_alpha 0.3 --beta 0.45

# CUDA_VISIBLE_DEVICES=0 python main.py --fin ../datasets/${dataname}/ --fout ../datasets/${output_folder}/drd_ideal --train_alg drd_ideal --pairwise 1 --lr 1e-4 --weight_decay 0 --batch_size 128  --topK ${TopK} --epoch 50  --randseed ${randseed} --drop_out 0.1 --eta ${eta}  --init_prob ${init_prob}  --alpha 0.18
