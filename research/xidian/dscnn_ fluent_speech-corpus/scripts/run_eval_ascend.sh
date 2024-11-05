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
# ===========================================================================

if [ $# != 2 ]
then
    echo "Usage: bash scripts/run_train_ascend.sh [EVAL_FEAT_DIR] [MODEL_DIR]"
    exit 1
fi

rm -rf eval
mkdir eval
cp -r ./src ./eval
cp -r ./scripts ./eval
cp ./*.py ./eval
cp ./*yaml ./eval
cd ./eval || exit
echo "start eval network"
python eval.py --eval_feat_dir=$1 --model_dir=$2 > eval.log 2>&1 &
cd ..
