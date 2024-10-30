python inference_old.py \
    --exp debug_old \
    --model_config config/model_config.json \
    --data_config config/data_config_old.json \
    --loss_measure moment_video \
    --load_dir results/debug_20241029_164559/0_model.pt \
    --local_batch_size 16 \
    --num_workers 8 