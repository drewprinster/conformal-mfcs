#!/bin/sh
#SBATCH -t 06:00:00 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=5G 
python ../run_SplitCP_MFCS_DesignExpts.py --dataset blue_protein --n_initial_all 64 --n_seed 500 --lmbdas 5 --p_split_train 0.5 --weight_depth_maxes 1 2 3 4 --prob_bound_inf 1.0
