#!/bin/sh
#SBATCH -t 36:00:00 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=10G 
python ../run_SplitCP_MFCS_ALexpts.py --dataset airfoil --n_steps 70 --n_initial_all 80 --n_seed 350 --lmbdas 10.0 --p_split_train 0.8 --noise_magnitude 0.05 --weight_depth_maxes 1 2 3 --initial_sampling_bias 3.0 --add_to_train_split_prob 0.5 --noise_level 0.05 --sigma_0 0.05 --prob_bound_inf 1.0
