model_name="bert4rec"
gpu_id=1
dataset="yelp"
mask_prob=0.6
seed=42

python main.py --dataset ${dataset} \
            --model_name ${model_name} \
            --hidden_size 64 \
            --max_len 200 \
            --gpu_id ${gpu_id} \
            --num_workers 8 \
            --mask_prob ${mask_prob} \
            --log \
            --num_train_epochs 1200 \
            --seed ${seed} \
            --aug_seq \
            --aug_seq_len 10 \
            --aug_file inter \
            --patience 200



