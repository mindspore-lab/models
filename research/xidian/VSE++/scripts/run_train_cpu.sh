#!/bin/bash
nohup python train.py --device "CPU"  > train_Ascend.log 2>&1 &
