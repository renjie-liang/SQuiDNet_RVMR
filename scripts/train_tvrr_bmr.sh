#!/usr/bin/env bash
python train_bmr.py \
    --exp top01_bmr \
    --model_config config/model_config.json \
    --data_config config/data_config_tvrr.json \
    --local_batch_size 8 \
    --local_eval_batch_size 4 \
    --loss_measure moment_video \
    --num_workers 8 \
    --eval_folds 0.2 \
    --lr 1e-5
