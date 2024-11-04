python inference_old_rvmr.py \
    --exp debug_old \
    --model_config config/model_config.json \
    --data_config config/data_config_tvrr.json \
    --loss_measure moment_video \
    --load_dir  results/top01_float32_20241030_000111/13_model.pt \
    --local_batch_size 16 \
    --num_workers 8 