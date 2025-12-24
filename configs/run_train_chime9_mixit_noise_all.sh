#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf "seanet-CHIME9-MixIT-training (all session pairs, noise) \n"
CUDA_VISIBLE_DEVICES=5 python ../main_chime9_mixit_noise_all.py \
--save_path exps/seanet_chime9_mixit_noise_all \
--data_list /net/midgar/work/nitsu/work/chime9/data/datalist_train \
--backbone seanet \
--n_cpu 4 \
--length 3 \
--batch_size 1 \
--max_epoch 150 \
--lr 0.00100 \
--alpha 0.10 \
--val_step 3 \
--lr_decay 0.97 \
--init_model /net/midgar/work/nitsu/work/chime9/SEANet/exps/seanet/model/model_0147.model

