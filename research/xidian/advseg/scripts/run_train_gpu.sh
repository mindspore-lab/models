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

## 确定传入的参数个数是否正确，3表示传入参数个数小于三个
#if [ $# -lt 3 ]
#then
#    echo "Usage: \
#          sh run_train_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATA_PATH]\
#          "
#exit 1
#fi
#
## 确定参数1使用卡的数目是否在1-8之间
#if [ $1 -lt 1 ] && [ $1 -gt 8 ]
#then
#    echo "error: DEVICE_NUM=$1 is not in (1-8)"
#exit 1
#fi
#
#export DEVICE_NUM=$1
#export RANK_SIZE=$1

export RANK_ID=0
export DEVICE_NUM=1


#EXEC_PATH=$(pwd)
DATA_PATH="/home/ma-user/work/data"
GTA5_PATH=${DATA_PATH}"/GTA5"
CITYSCAPES_PATH=${DATA_PATH}"/cityscapes"
time=$(date "+%Y-%m-%d-%H-%M")
checkpoint="./checkpoint/Multi_adv/"${time}
EXEC_PATH="$(pwd)/NPU_${DEVICE_NUM}P"
echo "the work path : $EXEC_PATH"
config_path="${EXEC_PATH}/advnet_config.yaml"
echo "config path is : ${config_path}"

#rm -rf $EXEC_PATH
mkdir $EXEC_PATH


cp -u ../*.py ../*.yaml $EXEC_PATH
cp -ru ../model_utils $EXEC_PATH
cp -ru ../modelarts $EXEC_PATH
cp -ru ../src $EXEC_PATH
cd $EXEC_PATH || exit

for((i=1;i<${RANK_SIZE};i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    mkdir -p ${checkpoint}"/device$i"
    python train_adv.py --snapshot_dir ${checkpoint} \
                    --data_dir ${GTA5_PATH} \
                    --data_dir_target ${CITYSCAPES_PATH} \
                    --restore_from /home/ma-user/work/model/Pretrain_DeeplabMulti.ckpt \
                    ----config_path ${config_path} \
                     >${checkpoint}"/device$i""/Train_device$i.log" &

done

export DEVICE_ID=0
export RANK_ID=0
echo "start training for device $RANK_ID"
mkdir -p ${checkpoint}"/device$RANK_ID"
python train_adv.py --snapshot_dir ${checkpoint} \
                --data_dir ${GTA5_PATH} \
                --data_dir_target ${CITYSCAPES_PATH} \
                --restore_from /home/ma-user/work/model/Pretrain_DeeplabMulti.ckpt \
                --config_path ${config_path} \
                --device_target "GPU"
#                >${checkpoint}"/device$i""/Train_device$i.log"


# 直接开始对抗训练

# 在source only基础上继续训练
#python train.py --snapshot-dir ./checkpoint/Multi_adv \
#                                     --lambda-seg 0.1 \
#                                     --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001 \
#                                     --data-dir /home/ma-user/work/data/GTA5 \
#                                     --data-dir-target /home/ma-user/work/data/cityscapes \
#                                     --input-size 1280,720 \
#                                     --input-size-target 1024,512 \
#                                     --restore-from ./checkpoint/source_only/2022-08-05-21-22-31/GTA5_best.ckpt \
#                                     --device ascend --batch-size 2

# 断点续训
#python train.py --snapshot-dir ./checkpoint/Multi_adv \
#                                     --lambda-seg 0.1 \
#                                     --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001 \
#                                     --data-dir /home/ma-user/work/data/GTA5 \
#                                     --data-dir-target /home/ma-user/work/data/cityscapes \
#                                     --input-size 1280,720 \
#                                     --input-size-target 1024,512 \
#                                     --restore-from model/Pretrain_DeeplabMulti.ckpt \
#                                     --device ascend --batch-size 2 \
#                                     --continue_train ./checkpoint/Multi_adv/best_model_checkpoint_41.8/GTA5_best.ckpt



# 直接开始对抗训练

# 在source only基础上继续训练
#python train.py --snapshot-dir ./checkpoint/Multi_adv \
#                                     --lambda-seg 0.1 \
#                                     --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001 \
#                                     --data-dir /home/ma-user/work/data/GTA5 \
#                                     --data-dir-target /home/ma-user/work/data/cityscapes \
#                                     --input-size 1280,720 \
#                                     --input-size-target 1024,512 \
#                                     --restore-from ./checkpoint/source_only/2022-08-05-21-22-31/GTA5_best.ckpt \
#                                     --device ascend --batch-size 2

# 断点续训
#python train.py --snapshot-dir ./checkpoint/Multi_adv \
#                                     --lambda-seg 0.1 \
#                                     --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001 \
#                                     --data-dir /home/ma-user/work/data/GTA5 \
#                                     --data-dir-target /home/ma-user/work/data/cityscapes \
#                                     --input-size 1280,720 \
#                                     --input-size-target 1024,512 \
#                                     --restore-from model/Pretrain_DeeplabMulti.ckpt \
#                                     --device ascend --batch-size 2 \
#                                     --continue_train ./checkpoint/Multi_adv/best_model_checkpoint_41.8/GTA5_best.ckpt



# if [ $# != 1 ] && [ $# != 2 ]
# then
#     echo "Usage: sh run_train.sh [RANK_TABLE_FILE] [cifar10|imagenet]"
# exit 1
# fi

# if [ ! -f $1 ]
# then
#     echo "error: RANK_TABLE_FILE=$1 is not a file"
# exit 1
# fi


# dataset_type='cifar10'
# if [ $# == 2 ]
# then
#     if [ $2 != "cifar10" ] && [ $2 != "imagenet" ]
#     then
#         echo "error: the selected dataset is neither cifar10 nor imagenet"
#     exit 1
#     fi
#     dataset_type=$2
# fi

# ulimit -u unlimited
# export DEVICE_NUM=8
# export RANK_SIZE=8
# PATH1=$(realpath $1)
# export RANK_TABLE_FILE=$PATH1
# echo "RANK_TABLE_FILE=${PATH1}"

# EXECUTE_PATH=$(pwd)
# config_path="${EXECUTE_PATH}/${dataset_type}_config.yaml"
# echo "config path is : ${config_path}"

# export SERVER_ID=0
# rank_start=$((DEVICE_NUM * SERVER_ID))


# cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
# avg=`expr $cpus \/ $DEVICE_NUM`
# gap=`expr $avg \- 1`

# for((i=0; i<${DEVICE_NUM}; i++))
# do
#     start=`expr $i \* $avg`
#     end=`expr $start \+ $gap`
#     cmdopt=$start"-"$end
#     export DEVICE_ID=$i
#     export RANK_ID=$((rank_start + i))
#     rm -rf ./train_parallel$i
#     mkdir ./train_parallel$i
#     cp -r ./src ./train_parallel$i
#     cp -r ./model_utils ./train_parallel$i
#     cp -r ./*.yaml ./train_parallel$i
#     cp ./train.py ./train_parallel$i
#     echo "start training for rank $RANK_ID, device $DEVICE_ID, $dataset_type"
#     cd ./train_parallel$i ||exit
#     env > env.log
#     taskset -c $cmdopt python train.py --config_path=$config_path --dataset_name=$dataset_type> log 2>&1 &
#     cd ..
# done
