mpirun -n 2 python -u train.py --model Nbeats --data M3C -c configs/nbeats/nbeats_train.yaml --seq_len 5 --pred_len 1 --do_train --distribute --device Ascend
mpirun -n 2 python -u train.py --model Nbeats --data M3C -c configs/nbeats/nbeats_train.yaml --seq_len 4 --pred_len 2 --do_train --distribute --device Ascend
mpirun -n 2 python -u train.py --model Nbeats --data M3C -c configs/nbeats/nbeats_train.yaml --seq_len 6 --pred_len 3 --do_train --distribute --device Ascend
mpirun -n 2 python -u train.py --model Nbeats --data M3C -c configs/nbeats/nbeats_train.yaml --seq_len 6 --pred_len 4 --do_train --distribute --device Ascend
mpirun -n 2 python -u train.py --model Nbeats --data M3C -c configs/nbeats/nbeats_train.yaml --seq_len 10 --pred_len 5 --do_train --distribute --device Ascend
mpirun -n 2 python -u train.py --model Nbeats --data M3C -c configs/nbeats/nbeats_train.yaml --seq_len 12 --pred_len 6 --do_train --distribute --device Ascend