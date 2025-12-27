#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf "seanet-CHIME9-MixIT-inference (session_00 all speakers) \n"
CUDA_VISIBLE_DEVICES=5 python ../main_inf_chime9_mixit_all.py \
--save_path exps/seanet_chime9_mixit_all \
--data_list /net/midgar/work/nitsu/work/chime9/data/datalist_train \
--backbone seanet \
--n_cpu 0 \
--init_model /net/midgar/work/nitsu/work/chime9/SEANet/configs/exps/seanet_chime9_mixit_all/model/model_0147.model \
--session_id session_00 \
--track track_00 \
--output_dir exps/seanet_chime9_mixit_all/inference/session_00

