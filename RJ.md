srun --partition=gpu --gres=gpu:1 --nodes=1 --cpus-per-task=1 --mem=32gb --time=01:00:00 --account=bianjiang --qos=bianjiang --reservation=bianjiang --pty bash -i
micromamba activate r2gen
sh scripts/train.sh 


/red/bianjiang/liang.renjie/RVMR/TVR-Ranking/data/features