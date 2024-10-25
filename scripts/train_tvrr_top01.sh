#!/usr/bin/env bash
python train.py \
    --exp debug \
    --model_config config/model_config.json \
    --data_config config/data_config_tvrr.json \
    --batch 16 \
    --max_vcmr_video 10 \
    --loss_measure moment_video \
    --num_workers 8 \
    --eval_folds 0.1