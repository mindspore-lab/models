#!/bin/bash

epochs=20  # 训练epoch
result_path="logs/"             # 保存log和权重的总路径
log_path="${result_path}log"
weight_path="${result_path}checkpoint"
datapath="./data/"
device_id=0

nohup python -u train.py --data_name f30k_precomp --data_path $datapath  --logger_name $log_path --model_name $weight_path --max_violation --bi_gru --agg_func=LogSumExp --cross_attn=t2i --lambda_lse=6 --lambda_softmax=9 --num_epochs=$epochs --lr_update=10 --learning_rate=.001 --device_target Ascend  --device_id $device_id  > train.log 2>&1 &
