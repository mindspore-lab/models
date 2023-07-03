#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

#bash scripts/run_infer_310.sh ./ascend310_infer/Advnet/GTA5_42.24.mindir ../data
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
    DEVICE_ID is optional, default value is zero"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}


model=$(get_real_path $1)
data_path="$(get_real_path $2)/Cityscapes"
#model=$1
#data_path="$2/Cityscapes"

device_id=0

if [ $# == 3 ]; then
    device_id=$3
fi

echo "The model path : $model"
echo "The data path  : $data_path"
echo "The device id  : $device_id"

#这部分应该不需要动############################### 具体根据环境设置，建议咨询
# version 1
export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then

    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export ASCEND_HOME=/usr/local/Ascend/latest/
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

function compile_app()
{
    cd ./ascend310_infer || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log

    if [ $? -ne 0 ]; then
        echo "compile app code failed"
        exit 1
    fi
    cd - || exit
}
#
###################################################
#
function preprocess_data()
 # 根据需求决定对preprocess.py进行适当调整
{
    # python ./preprocess.py --data_dir_target=/home/lcy/data/Cityscapes --config_path="./advnet_config.yaml" --output_path="./preprocess_Result"
    output_path="./preprocess_Result"
    if [ -d $output_path ]; then
        rm -rf $output_path
    fi
    mkdir $output_path
    config_path="./advnet_config.yaml"
    python ./preprocess.py --data_dir_target ${data_path} --config_path=$config_path --output_path=$output_path
}


function infer()
{
    predict_Files="./result_Files"
    time_Result="./time_Result"
    if [ -d $predict_Files ]; then
        rm -rf $predict_Files
    fi
     if [ -d $time_Result ]; then
        rm -rf $time_Result
    fi
    mkdir $predict_Files
    mkdir $time_Result
    ../ascend310_infer/out/main --model_path=$model --dataset_path="$output_path/img_data" --device_id=$device_id &> "${time_Result}/infer.log"

    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
}

function postprocess()
{
    # python ./postprocess.py  --data_dir_target=/home/lcy/data/Cityscapes --prefict_path=./result_Files
    python ./postprocess.py  --data_dir_target=${data_path} --prefict_path=$predict_Files

    if [ $? -ne 0 ]; then
        echo "calculate accuracy failed"
        exit 1
    fi
}
##
##if [ "x${dataset}" == "xcifar10" ] || [ "x${dataset}" == "xCifar10" ]; then
##    preprocess_data
##    data_path=./preprocess_Result/img_data
##fi

#compile_app
preprocess_data
infer
postprocess
