#!/bin/sh
export METHOD=MME
export NET=resnet34
export D=$1
export S=$2
export T=$3
export N=$4
python eval.py \
    --method $METHOD \
    --device_id $D \
    --dataset multi \
    --source $S \
    --target $T \
    --net $NET \
    --num $N \ 
    --save_check \ 