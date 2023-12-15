export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RANK_SIZE=$1
export CUDA_NUM=$1

if (($1 == 1))  
then
    python ./test_pipeline.py --rank_size 1 --task $2 --model $3 --dataset $4 
else
    rm -rf device0
    mkdir device0
    cp -a ./config ./data ./evaluator ./executor ./kernel_meta ./log ./model ./pipeline ./rank_0 ./raw_data ./utils ./test_pipeline.py  ./device0
    cd ./device0
    echo "start training"
    mpirun --allow-run-as-root -n ${RANK_SIZE} python3 ./test_pipeline.py --rank_size ${RANK_SIZE} --task $2 --model $3 --dataset $4 
fi