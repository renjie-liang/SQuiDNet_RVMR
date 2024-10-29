#!/usr/bin/env bash
python train.py \
    --exp debug \
    --model_config config/model_config.json \
    --data_config config/data_config_tvrr.json \
    --local_batch_size 16 \
    --num_workers 8 \
    --eval_folds 10