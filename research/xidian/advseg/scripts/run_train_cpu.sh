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

# 确定传入的参数个数是否正确，3表示传入参数个数小于三个
if [ $# -lt 2 ]
then
    echo "Usage: \
          sh run_train_gpu.sh [DATA_PATH]\
          "
exit 1
fi

# 确定参数1使用卡的数目是否在1-8之间
if [ $1 -lt 1 ] && [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in (1-8)"
exit 1
fi

export DEVICE_NUM=$1


#EXEC_PATH=$(pwd)
DATA_PATH="/home/ma-user/work/data"
GTA5_PATH=${DATA_PATH}"/GTA5"
CITYSCAPES_PATH=${DATA_PATH}"/cityscapes"
time=$(date "+%Y-%m-%d-%H-%M")
checkpoint="./checkpoint/Multi_adv/"${time}
EXEC_PATH="$(pwd)/CPU"
echo "the work path : $EXEC_PATH"
config_path="${EXEC_PATH}/advnet_config.yaml"
echo "config path is : ${config_path}"


mkdir $EXEC_PATH


cp -u ../*.py ../*.yaml $EXEC_PATH
cp -ru ../model_utils $EXEC_PATH
cp -ru ../modelarts $EXEC_PATH
cp -ru ../src $EXEC_PATH
cd $EXEC_PATH || exit

python train_adv.py --snapshot_dir ${checkpoint} \
                --data_dir ${GTA5_PATH} \
                --data_dir_target ${CITYSCAPES_PATH} \
                --restore_from /home/ma-user/work/model/Pretrain_DeeplabMulti.ckpt \
                --config_path ${config_path} \
                --device_target "CPU"
#                >${checkpoint}"/device$i""/Train_device$i.log"

