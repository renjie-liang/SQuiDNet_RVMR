#!/usr/bin/env bash
python train.py \
    --exp moment_video \
    --model_config config/model_config.json \
    --data_config config/data_config.json \
    --batch 16 \
    --max_vcmr_video 10 \
    --loss_measure moment_video \
    --num_workers 1\
    ${@:1}
