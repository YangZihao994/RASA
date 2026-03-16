#!/bin/bash

datapath='/gaoxieping/yzh/Dataset/BraTs_2023_GLI_npy'
savepath='./checkpoints'
 

cd "/gaoxieping/yzh/Code/RASA"

/gaoxieping/miniconda3/envs/yzh/bin/python train.py \
    --batch_size=12 \
    --data_root $datapath \
    --save_dir $savepath \
    --num_epochs 100 \
    --num_workers 8
