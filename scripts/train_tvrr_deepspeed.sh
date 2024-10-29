#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6,7 deepspeed --num_gpus=1 train_deepspeed.py \
                    --exp debug \
                    --model_config config/model_config.json \
                    --data_config config/data_config_tvrr.json \
                    --deepspeed_config config/deepspeed_config.json \
                    --num_workers 8 