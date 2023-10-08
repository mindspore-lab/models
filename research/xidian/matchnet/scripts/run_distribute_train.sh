#!/bin/bash
# Copyright 2023 Xidian University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 7 ] && [ $# != 8 ]
then
    echo "Using: bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK] [DATASET_NAME] [EVAL_DATASET_NAME]"
    echo "or"
    echo "Using: bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]   [DATASET_NAME] [EVAL_DATASET_NAME]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH1=$(get_real_path $1)    # rank_table_file
PATH2=$(get_real_path $2)    # dataset_path
PATH3=$(get_real_path $3)    # train_output_path

if [ ! -f $PATH1 ]
then
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file."
    exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: DATASET_PATH=$PATH2 is not a directory."
    exit 1
fi

if [ ! -d $PATH3 ]
then
    echo "error: TRAIN_OUTPUT_PATH=$PATH3 is not a directory."
fi

if [ $4 != "None" ]
then
    PATH4=$(get_real_path $4)    # pretrained or resume ckpt_path
    if [ ! -f $PATH4 ]
    then
        echo "error: CHECKPOINT_PATH=$PATH4 is not a file."
        exit 1
    fi
else
    PATH4=$4
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ./train.py ./train_parallel$i
    cp -r ./src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "Start training for rank $RANK_ID, device $DEVICE_ID."
    env > env.log
    if [ $# == 7 ]
    then
        if [ -f $PATH4 ]
        then
            python train.py --data_url $PATH2 --train_url $PATH3 --checkpoint_url $PATH4 --modelarts False --run_distribute True --eval $5 --dataset $6 --evalset $7 &> log &
        else
            python train.py --data_url $PATH2 --train_url $PATH3 --modelarts False --run_distribute True --eval $5 --dataset $6 --evalset $7 &> log &
        fi
    elif [ $# == 8 ]
    then
        if [ -f $PATH4 ]
        then
            python train.py --data_url $PATH2 --train_url $PATH3 --checkpoint_url $PATH4 --modelarts False --run_distribute True --begin_epoch $5 --eval $6  --dataset $7 --evalset $8 &> log &
        else
            python train.py --data_url $PATH2 --train_url $PATH3 --modelarts False --run_distribute True --begin_epoch $5 --eval $6  --dataset $7 --evalset $8 &> log &
        fi
    fi
    cd ..
done
