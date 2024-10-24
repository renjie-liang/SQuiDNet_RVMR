#!/bin/bash
#SBATCH --partition=gpu                # Partition to use (GPU partition)
#SBATCH --gres=gpu:1                   # Number of GPUs (1)
#SBATCH --nodes=1                      # Number of nodes (1)
#SBATCH --cpus-per-task=8            # Number of CPUs per task (18)
#SBATCH --mem=100gb                    # Total memory (512 GB)
#SBATCH --time=48:00:00               # Time limit (48 hours)
#SBATCH --account=bianjiang            # Account name (bianjiang)
#SBATCH --qos=bianjiang                # Quality of Service (bianjiang)
#SBATCH --reservation=bianjiang        # Reservation (bianjiang)
#SBATCH --job-name=renjie_job          # Name of your job            
#SBATCH --output=work_dirs/slurm_logs/slurm-1gpus-%j.out


eval "$(micromamba shell hook --shell bash)"

micromamba activate squid

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
