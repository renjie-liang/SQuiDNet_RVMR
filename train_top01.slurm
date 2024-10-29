#!/bin/bash
#SBATCH --partition=gpu                # Partition to use (GPU partition)
#SBATCH --gres=gpu:1                   # Number of GPUs (1)
#SBATCH --nodes=1                      # Number of nodes (1)
#SBATCH --cpus-per-task=32            # Number of CPUs per task (18)
#SBATCH --mem=768gb                    # Total memory (512 GB)
#SBATCH --time=300:00:00               # Time limit (48 hours)
#SBATCH --account=bianjiang            # Account name (bianjiang)
#SBATCH --qos=bianjiang                # Quality of Service (bianjiang)
#SBATCH --reservation=bianjiang        # Reservation (bianjiang)
#SBATCH --job-name=top01                # Name of your job            
#SBATCH --output=work_dirs/slurm-%j.out


eval "$(micromamba shell hook --shell bash)"
export CUDA_VISIBLE_DEVICES==5

micromamba activate r2gen
deepspeed --num_gpus=1 train_deepspeed.py \
        --exp top_01 \
        --model_config config/model_config.json \
        --data_config config/data_config_tvrr.json \
        --deepspeed_config config/deepspeed_config.json \
        --num_workers 8 