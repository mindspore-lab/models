echo "Please run the script as: "
echo "bash scripts/run_distribute_train.sh DEVICE_NUM DATASET_PATH"
echo "for example: bash scripts/run_distribute_train.sh 8 /dataset_path"
echo "After running the script, the network runs in the background, The log will be generated in log/output.log"


export RANK_SIZE=$1
DATA_URL=$2

rm -rf log
mkdir ./log
cp *.py ./log
cp *.yaml ./log
cp -r src ./log
cd ./log || exit
env > env.log
mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
  python -u train.py \
    --dataset_path=$DATA_URL \
    --ckpt_path="./" \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='GPU' \
    --do_eval=True > output.log 2>&1 &
