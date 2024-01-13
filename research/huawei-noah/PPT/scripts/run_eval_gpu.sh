#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

SCRIPT_DIR=$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)
echo "SCRIPT_DIR=$SCRIPT_DIR"

display_usage() {
    echo -e "Usage: $0 CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]"
    echo "Important! Extra args must be the last argument."
#    echo -e "$help"
}

# Check if help in CLI arguments
for arg in "$@"
do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    display_usage
    exit 0
  fi
done

# Check if there are enough arguments
# If yes, parse the first three
if [[ $# -lt 1 ]]; then
    echo "Not enough arguments"
    exit 1
else
    CONFIG_PATH="$1"
    shift 1
fi

DEVICE_ID="0"
EXTRA=""

# Parse remain arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint)
            if [[ "$#" -lt 2 ]]; then
                echo "No CHECKPOINT option"
                exit 1
            else
                CHECKPOINT="$2";
                shift
            fi ;;
        --device)
            if [[ "$#" -lt 2 ]]; then
                echo "No GPU option"
                exit 1
            else
                DEVICE_ID="$2";
                shift
            fi ;;
        --extra)
            if [[ "$#" -lt 2 ]]; then
                echo "No EXTRA option"
                exit 1
            else
                shift;
                EXTRA="$*";
                break
            fi ;;
        *) echo "Unknown option: '$1'"; exit 1 ;;
    esac
    shift
done

EVAL_SCRIPT_DIR="$SCRIPT_DIR/.."

# If variable CHECKPOINT is empty then evaluation can not be performed.
# Otherwise, run evaluation.
echo "Start evaluation for device $DEVICE_ID"
if [ -z "$CHECKPOINT" ]; then
    echo "Error! Expected --checkpoint option. "
    exit 1
fi

python3 "$EVAL_SCRIPT_DIR/eval.py" --config "$CONFIG_PATH" --device_id $DEVICE_ID --finetune "$CHECKPOINT" $EXTRA  2>&1 | tee eval.log
