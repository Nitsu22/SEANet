#!/bin/bash

printf "seanet-CHIME9-MixIT-training \n"
CUDA_VISIBLE_DEVICES=0 python ../main_chime9_mixit.py \
--save_path exps/seanet_chime9_mixit \
--data_list /net/midgar/work/nitsu/work/chime9/data/datalist_train \
--backbone seanet \
--n_cpu 12 \
--length 4 \
--batch_size 1 \
--max_epoch 150 \
--lr 0.00100 \
--alpha 1.0 \
--val_step 3 \
--lr_decay 0.97

