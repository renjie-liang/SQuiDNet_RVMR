#!/bin/bash
#SBATCH --partition=gpu                # Partition to use (GPU partition)
#SBATCH --gres=gpu:1                   # Number of GPUs (1)
#SBATCH --nodes=1                      # Number of nodes (1)
#SBATCH --cpus-per-task=16            # Number of CPUs per task (18)
#SBATCH --mem=200gb                    # Total memory (512 GB)
#SBATCH --time=300:00:00               # Time limit (48 hours)
#SBATCH --account=bianjiang            # Account name (bianjiang)
#SBATCH --qos=bianjiang                # Quality of Service (bianjiang)
#SBATCH --reservation=bianjiang        # Reservation (bianjiang)
#SBATCH --job-name=renjie_job          # Name of your job            
#SBATCH --output=work_dirs/slurm-1gpus-%j.out


eval "$(micromamba shell hook --shell bash)"
# export CUDA_VISIBLE_DEVICES==5

micromamba activate r2gen

python train.py \
    --exp top01_vcmr \
    --model_config config/model_config.json \
    --data_config config/data_config_tvrr.json \
    --local_batch_size 16 \
    --loss_measure moment_video \
    --num_workers 8 \
    --eval_folds 1 \
    --lr 1e-5