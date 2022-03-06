#!/bin/bash

for((i=2; i<=7; i++))
do
    ((lr_mag=${i}))
    python train_2.py --lr 2  --batch_size 16 --sd_fac 5 --lr_mag ${lr_mag} > log/AlbertLSTM_log.lr_mag${i}
done
