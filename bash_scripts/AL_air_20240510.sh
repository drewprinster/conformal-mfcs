#!/bin/sh
#SBATCH -t 2:00:00 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=100G 
python ../run_SplitCP_MultistepFCS_ALexpts_20240510.py --dataset airfoil --n_steps 10 --n_train_initial 80 --lmbdas 10.0 --p_split_train 0.8 --noise_magnitude 0.05 --weight_adj_depth_maxes 1 2 3 4 5 6 --initial_sampling_bias 3.0 --add_to_train_split_prob 0.5 --noise_level 0.05 --n_seed 1 --sigma_0 0.05 --beta 1.0 --aci_step_size 0.75
