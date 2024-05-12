#!/bin/sh
#SBATCH -t 06:00:00 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=5G 
python ../run_SplitCP_MultistepFCS_DesignExpts_20240510.py --dataset blue_protein --n_train_initial 64  --lmbdas 5 --p_split_train 0.5 --weight_adj_depth_maxes 1 2 3 4 --n_seed 500
