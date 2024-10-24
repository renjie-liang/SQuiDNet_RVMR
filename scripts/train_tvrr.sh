#!/usr/bin/env bash
python train.py \
    --exp moment_video \
    --model_config config/model_config.json \
    --data_config config/data_config_tvrr.json \
    --batch 32 \
    --eval_query_batch 5 \
    --task VCMR \
    --eval_tasks VCMR SVMR VR \
    --max_vcmr_video 10 \
    --loss_measure moment_video \
    --num_workers 8 \
    --exp_id debug \
    --eval_folds 1