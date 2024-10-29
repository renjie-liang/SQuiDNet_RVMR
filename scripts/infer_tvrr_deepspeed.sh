CUDA_VISIBLE_DEVICES=5 deepspeed --num_gpus=1 inference_deepspeed.py \
                    --exp debug \
                    --model_config config/model_config.json \
                    --data_config config/data_config_tvrr.json \
                    --deepspeed_config config/deepspeed_config.json \
                    --max_vcmr_video 10 \
                    --loss_measure moment_video \
                    --load_dir results/debug_20241029_004116 \
                    --num_workers 16 \