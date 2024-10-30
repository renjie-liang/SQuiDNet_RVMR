#!/usr/bin/env bash
python train.py \
    --exp top01_vcmr \
    --model_config config/model_config.json \
    --data_config config/data_config_tvrr.json \
    --local_batch_size 16 \
    --loss_measure moment_video \
    --num_workers 8 \
    --eval_folds 1 \
    --lr 1e-5