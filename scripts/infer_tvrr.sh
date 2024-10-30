python inference.py \
    --exp debug \
    --model_config config/model_config.json \
    --data_config config/data_config_tvrr.json \
    --load_dir results/debug_20241029_164559/1_model.pt \
    --local_batch_size 16 \
    --num_workers 8 