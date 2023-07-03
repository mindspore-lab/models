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

# 确定传入的参数个数是否正确，2表示传入参数个数小于两个
if [ $# -lt 3 ]
then
    echo "Usage: \
          sh run_train_gpu.sh [DEVICE_NUM] [DATA_PATH]\
          "
exit 1
fi

#
# 确定参数1使用卡的数目是否在1-8之间
if [ $1 -lt 1 ] && [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in (1-8)"
exit 1
fi

export DEVICE_NUM=$1
export RANK_SIZE=$1



#EXEC_PATH=$(pwd)
time=$(date "+%Y-%m-%d-%H-%M")
checkpoint="./checkpoint/eval/Multi_adv/"${time}
EXEC_PATH="$(pwd)/NPU_${DEVICE_NUM}P_EVAL"
echo "the work path : $EXEC_PATH"
config_path="${EXEC_PATH}/advnet_config.yaml"
echo "config path is : ${config_path}"
RESTORE_FROM=$1
echo "The Test checkpoint ckpt is : $RESTORE_FROM"

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
    python eval_adv.py --config_path ${config_path} \
                       --restore_from $RESTORE_FROM \
                        >${checkpoint}"/device$i""/Train_device$i.log" &

done

export DEVICE_ID=0
export RANK_ID=0
echo "start training for device $RANK_ID"
mkdir -p ${checkpoint}"/device$RANK_ID"
python eval_adv.py --config_path ${config_path} \
                       --restore_from $RESTORE_FROM \
                       --set val \
                        >${checkpoint}"/device$i""/Train_device$i.log" &

