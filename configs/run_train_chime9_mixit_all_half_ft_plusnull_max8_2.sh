#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf "seanet-CHIME9-MixIT-training (all session pairs with plusnull, max8 speakers, max2 session3) \n"
# Use CUDA_VISIBLE_DEVICES from environment if set, otherwise default to 0
CUDA_VISIBLE_DEVICES=6 python ../main_chime9_mixit_half_all_plusnull_max8-2.py \
--save_path exps/seanet_chime9_mixit_all_half_ft_plusnull_max8_2 \
--data_list /net/midgar/work/nitsu/work/chime9/data/datalist_train \
--backbone seanet \
--n_cpu 4 \
--length 4 \
--batch_size 1 \
--max_epoch 150 \
--lr 0.00100 \
--alpha 1.0 \
--val_step 3 \
--lr_decay 0.97 \
--init_model /net/midgar/work/nitsu/work/chime9/SEANet/exps/seanet/model/model_0147.model
