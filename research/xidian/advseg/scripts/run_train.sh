#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
#if [ $# -lt 2 ]
#then
#    echo "Usage: \
#          sh run_train_gpu.sh [DEVICE_NUM] [DATA_PATH]\
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
export DEVICE_NUM=8
export RANK_SIZE=${DEVICE_NUM}

DATA_PATH="/tmp/data"
#DATA_PATH=$2
GTA5_PATH="${DATA_PATH}/GTA5"
CITYSCAPES_PATH="${DATA_PATH}/Cityscapes"


LOG_PATH="./log"

EXEC_PATH="$(pwd)/NPU_${DEVICE_NUM}P"
echo "the work path : $EXEC_PATH"
CONFIG_PATH="${EXEC_PATH}/advnet_config.yaml"
#CONFIG_PATH="./advnet_config.yaml"
echo "config path is : ${CONFIG_PATH}"
#
mkdir $EXEC_PATH

cp -u ./*.py ./*.yaml ./*.json $EXEC_PATH
cp -rfu ./model_utils ./modelarts ./src $EXEC_PATH
cd $EXEC_PATH || exit

RANK_TABLE_FILE="$(pwd)/rank_table_8pcs.json"
export RANK_TABLE_FILE
echo "the rank table path : $RANK_TABLE_FILE"

for((i=0;i<${RANK_SIZE};i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $RANK_ID"
    rm -rf ./rank_$RANK_ID
    # mkdir -p ${checkpoint}"/device$i"
    mkdir -p ${LOG_PATH}"/device$i"
    python train_adv.py --data_dir ${GTA5_PATH} \
                        --data_dir_target ${CITYSCAPES_PATH} \
                        --config_path ${CONFIG_PATH} \
                        >${LOG_PATH}"/device$i/Train_device$i.log" 2>&1 &
#    mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout pytest -s -v ./train_adv.py # > train.log 2>&1 &
#    pytest ./train_adv.py # \ > ${LOG_PATH}"/device$i/Train_device$i.log" 2>&1 &
done

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi

#export DEVICE_ID=0
#export RANK_ID=0
#echo "start training for device $RANK_ID"
#rm -rf ./rank_$RANK_ID

#mkdir -p ${checkpoint}"/device$RANK_ID"


#python train_adv.py --config_path ${CONFIG_PATH}

#python train_adv.py --snapshot_dir ${checkpoint} \
#                --data_dir ${GTA5_PATH} \
#                --data_dir_target ${CITYSCAPES_PATH} \
#                --restore_from /home/ma-user/work/model/Pretrain_DeeplabMulti.ckpt \
#                --config_path ${config_path} \
#                >${checkpoint}"/device$i""/Train_device$i.log"

